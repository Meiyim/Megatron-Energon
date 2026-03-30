# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import inspect
import linecache
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensorBase

# For the watch_iter type
T = TypeVar("T")

# Maximum length of a single object string to print.
PRINT_LOCAL_MAX_LENGTH = 250

# Memory threshold: trigger OOM dump when available memory drops below this (GB).
# Set via ENERGON_MEM_AVAIL_THRESHOLD_GB. Default 200 GB.
_MEM_AVAIL_THRESHOLD_GB = float(os.environ.get("ENERGON_MEM_AVAIL_THRESHOLD_GB", "600"))
# How often the daemon thread checks memory (seconds).
_MEM_CHECK_INTERVAL = float(os.environ.get("ENERGON_MEM_CHECK_INTERVAL", "3600"))


class Watchdog:
    """
    A watchdog timer that:
      - can be 'enabled' or 'disabled' by presence/absence of a deadline,
      - resets automatically when 'enable()' is called,
      - can be used as a context manager,
      - can wrap an iterator to watch only the time for 'next()' calls,
      - attempts a two-phase shutdown on callback error:
         1) sys.exit(1) for graceful,
         2) if still alive after 10s, os._exit(1).
    """

    def __init__(
        self,
        timeout: float,
        initial_timeout: Optional[float] = None,
        callback: Optional[Callable[[], None]] = None,
        dump_stacks: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            timeout: Number of seconds before the watchdog fires if not reset/disabled.
            initial_timeout: Number of seconds before the watchdog fires in the first iteration.
            callback: Optional function to call upon timeout.
            dump_stacks: If True, print full stack traces for all threads on timeout (except watchdog's own thread).
            enabled: If False, watchdog starts disabled until enable() is called.
        """
        self._timeout = timeout
        self._initial_timeout = initial_timeout
        self._callback = callback
        self._dump_stacks = dump_stacks
        self._is_first_iteration = True

        # If _deadline is None, the watchdog is disabled.
        # Otherwise, _deadline = time.time() + _timeout if enabled.
        if enabled:
            self._deadline: Optional[float] = perf_counter() + self._get_next_timeout()
        else:
            self._deadline = None

        self._stop = False  # signals permanent shutdown (finish)
        self._mem_dumped = False  # only dump once per watchdog instance

        # Condition variable to manage state changes
        self._cv = threading.Condition()
        # Background thread (daemon) that monitors timeouts
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _get_next_timeout(self) -> float:
        if self._is_first_iteration:
            self._is_first_iteration = False
            return self._initial_timeout if self._initial_timeout is not None else self._timeout
        else:
            return self._timeout

    def _worker(self) -> None:
        """
        Background daemon thread that:
        1) Checks if the watchdog deadline has expired (timeout).
        2) Periodically checks available memory and dumps diagnostics if low.
        """
        while True:
            with self._cv:
                if self._stop:
                    return

                # --- Memory check (runs every iteration, ~every _MEM_CHECK_INTERVAL seconds) ---
                if not self._mem_dumped and _MEM_AVAIL_THRESHOLD_GB > 0:
                    try:
                        avail = get_mem_available_gb()
                        rss = get_rss_gb()
                        if avail < _MEM_AVAIL_THRESHOLD_GB:
                            self._mem_dumped = True
                            print(
                                f"[MEM_WATCHDOG] pid={os.getpid()} MEMORY LOW: "
                                f"avail={avail:.1f}G < threshold={_MEM_AVAIL_THRESHOLD_GB:.0f}G, "
                                f"rss={rss:.1f}G  — dumping stacks & memory stats",
                                flush=True,
                            )
                            self._print_all_thread_stacks(skip_thread_id=threading.get_ident())
                            self._print_memory_stats()
                    except Exception:
                        pass

                # --- Watchdog timeout check ---
                if self._deadline is None:
                    self._cv.wait(timeout=_MEM_CHECK_INTERVAL)
                    continue

                remaining = self._deadline - perf_counter()
                if remaining <= 0:
                    self._on_timeout()
                    return
                else:
                    self._cv.wait(timeout=min(remaining, _MEM_CHECK_INTERVAL))

    @staticmethod
    def _print_memory_stats() -> None:
        """Dump process and GPU memory stats to help diagnose OOM.
        Writes to both stdout and a per-PID file to avoid interleaving."""
        pid = os.getpid()
        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_dir = Path("/tmp/watchdog_dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"watchdog_mem_pid{pid}_{ts}.txt"

        lines = []
        lines.append("=" * 60)
        lines.append(f"Watchdog Memory Report  pid={pid}  ts={ts}")
        lines.append("=" * 60)

        # 1) Process RSS / VMS from /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:", "VmSwap:", "Threads:")):
                        lines.append(f"  {line.rstrip()}")
        except Exception:
            # Fallback: resource module
            ru = resource.getrusage(resource.RUSAGE_SELF)
            lines.append(f"  maxrss (resource): {ru.ru_maxrss} KB")

        # 2) Python GC stats
        gc_stats = gc.get_stats()
        lines.append(f"  gc generations: {[{'collections': s['collections'], 'collected': s['collected'], 'uncollectable': s['uncollectable']} for s in gc_stats]}")
        lines.append(f"  gc tracked objects: {len(gc.get_objects())}")

        # 3) GPU memory (all visible devices)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    max_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                    max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
                    lines.append(
                        f"  GPU {i}: alloc={alloc:.2f}G  reserved={reserved:.2f}G  "
                        f"max_alloc={max_alloc:.2f}G  max_reserved={max_reserved:.2f}G"
                    )
                except Exception as e:
                    lines.append(f"  GPU {i}: <error: {e}>")

        # 4) Top 10 object types by count (lightweight)
        try:
            import collections as _collections
            type_counts = _collections.Counter(type(o).__name__ for o in gc.get_objects())
            lines.append("  Top 10 GC object types by count:")
            for name, count in type_counts.most_common(10):
                lines.append(f"    {name}: {count}")
        except Exception:
            pass

        # 5) PIL images alive in GC
        try:
            from PIL.Image import Image as _PILImage
            _pil_objs = [(o.size, o.mode) for o in gc.get_objects() if isinstance(o, _PILImage)]
            _pil_mb = sum(w * h * len(m) / (1024 ** 2) for (w, h), m in _pil_objs)
            lines.append(
                f"  PIL images in GC: count={len(_pil_objs)}, est_ram={_pil_mb:.1f}MB"
            )
            if _pil_objs:
                import collections as _collections
                _size_counts = _collections.Counter((w, h) for (w, h), _ in _pil_objs)
                for (w, h), cnt in _size_counts.most_common(10):
                    lines.append(f"    {w}x{h}: {cnt} image(s)")
        except Exception:
            pass

        # 6) Large CPU tensors (likely pixel-value buffers)
        try:
            _large = [
                o for o in gc.get_objects()
                if isinstance(o, torch.Tensor) and not o.is_cuda and o.numel() > 100_000
            ]
            _large.sort(key=lambda t: t.numel(), reverse=True)
            _tensor_mb = sum(t.numel() * t.element_size() for t in _large) / (1024 ** 2)
            lines.append(
                f"  Large CPU tensors (>100k elem): count={len(_large)}, total={_tensor_mb:.1f}MB"
            )
            for t in _large[:10]:
                lines.append(
                    f"    shape={list(t.shape)} dtype={t.dtype} "
                    f"{t.numel() * t.element_size() / (1024**2):.1f}MB"
                )
        except Exception:
            pass

        # 7) Open file descriptors
        try:
            fd_count = len(os.listdir("/proc/self/fd"))
            lines.append(f"  Open FDs: {fd_count}")
        except Exception:
            pass

        lines.append("=" * 60)
        report = "\n".join(lines)

        # Write to file (atomic per-PID, no conflict)
        try:
            with open(dump_path, "w") as f:
                f.write(report + "\n")
            print(f"Watchdog memory report written to {dump_path}", flush=True)
        except Exception:
            pass

        # Also print to stdout
        print(report, flush=True)

    def _on_timeout(self) -> None:
        """
        Called exactly once if the watchdog times out.
        1) Optionally dumps stacks,
        1.5) Dump memory stats (RSS, GPU, GC) for OOM diagnosis,
        2) Calls user callback,
        3) If callback raises an error,
           - print traceback,
           - sys.exit(1),
           - fallback to os._exit(1) after 10s if process not terminated.
        """
        watchdog_thread_id = threading.get_ident()

        # 1) Dump stacks if requested
        if self._dump_stacks:
            print("Watchdog triggered: Dumping thread stacks")
            self._print_all_thread_stacks(skip_thread_id=watchdog_thread_id)

        # 1.5) Always dump memory stats for OOM diagnosis
        try:
            self._print_memory_stats()
        except Exception:
            traceback.print_exc()

        # 2) Call user callback
        if self._callback:
            try:
                self._callback()
            except Exception:
                # Print the traceback
                traceback.print_exc()

                # Start a background kill-switch after 10 seconds
                def force_exit_after_delay() -> None:
                    time.sleep(10)
                    os._exit(1)

                killer = threading.Thread(target=force_exit_after_delay, daemon=True)
                killer.start()

                # Attempt graceful shutdown
                sys.exit(1)

    def _print_all_thread_stacks(self, skip_thread_id: Optional[int] = None) -> None:
        """
        Dump stacks of all threads in a style reminiscent of py-spy, from
        innermost (current) to outermost. Skip the watchdog's own thread if given.

        Args:
            skip_thread_id: If given, skip this thread's stack.
        """

        frames = sys._current_frames()  # thread_id -> frame
        # We gather known threads to print their names
        all_threads = {t.ident: t for t in threading.enumerate()}

        for thread_id, frame in frames.items():
            if skip_thread_id is not None and thread_id == skip_thread_id:
                continue

            thread = all_threads.get(thread_id)
            thread_name = thread.name if thread else f"Unknown-{thread_id}"
            print(f'Thread {thread_id}: "{thread_name}"')

            # Build the stack from current (innermost) to outermost
            stack_frames = []
            f = frame
            while f is not None:
                stack_frames.append(f)
                f = f.f_back

            for fr in stack_frames:
                code = fr.f_code
                func_name = code.co_name
                filename = code.co_filename
                lineno = fr.f_lineno

                print(f"    {func_name} ({filename}:{lineno})")

                # Attempt to read the actual line of source
                line = linecache.getline(filename, lineno).rstrip()
                if line:
                    print(f"        > {line}")

                # Show arguments and locals
                arg_info = inspect.getargvalues(fr)
                arg_names = arg_info.args
                varargs = arg_info.varargs
                varkw = arg_info.keywords
                local_vars = arg_info.locals

                # Separate out the arguments
                arg_dict = {}
                for arg in arg_names:
                    if arg in local_vars:
                        arg_dict[arg] = local_vars[arg]
                if varargs and varargs in local_vars:
                    arg_dict["*" + varargs] = local_vars[varargs]
                if varkw and varkw in local_vars:
                    arg_dict["**" + varkw] = local_vars[varkw]

                if arg_dict:
                    print("        Arguments:")
                    for k, v in arg_dict.items():
                        print(f"            {k}: {repr_short(v)}")

                other_locals = {k: v for k, v in local_vars.items() if k not in arg_dict}
                if other_locals:
                    print("        Locals:")
                    for k, v in other_locals.items():
                        print(f"            {k}: {repr_short(v)}")

            print(flush=True)

    def reset(self) -> None:
        """
        Reset the watchdog timer (push out deadline by `timeout` seconds),
        but only if currently enabled (i.e., _deadline is not None).
        """
        with self._cv:
            if self._deadline is not None:
                self._deadline = perf_counter() + self._timeout
                self._cv.notify()

    def enable(self) -> None:
        """
        Enable (or re-enable) the watchdog. Always resets the deadline to
        `time.time() + timeout`.
        """
        with self._cv:
            self._deadline = perf_counter() + self._get_next_timeout()
            self._cv.notify()

    def disable(self) -> None:
        """
        Disable the watchdog (no timeout will fire until re-enabled).
        """
        with self._cv:
            self._deadline = None
            self._cv.notify()

    def finish(self) -> None:
        """
        Permanently stop the watchdog thread and disarm the timer.
        After calling finish(), you cannot re-enable this watchdog.
        """
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._worker_thread.join()

    def __enter__(self) -> "Watchdog":
        # If currently disabled, calling enable() will also reset the timer
        if self._deadline is None:
            self.enable()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # End the watchdog on context exit
        self.finish()

    def watch_iter(self, iterable: Iterable[T]) -> Iterator[T]:
        """
        Wrap an iterable so that each 'next()' call is watched by the watchdog,
        but the time in between iterations is not watched. Usage:

            wd = Watchdog(timeout=3, enabled=False)
            for item in wd.watch_iter(generator()):
                # processing item not timed by the watchdog
                pass

        This pattern:
          - enable() -> sets/extends deadline
          - next(...) -> measured portion
          - disable() -> stops timer

        Args:
            iterable: The iterable to wrap and watch.

        Returns:
            An iterator that wraps the input iterable and watches for timeouts.
        """
        try:
            self.enable()
            for item in iterable:
                self.disable()
                yield item
                self.enable()
        finally:
            self.disable()


def repr_short(obj: Any) -> str:
    """
    Return a short repr of an object.
    """
    if isinstance(obj, torch.Tensor):
        if isinstance(obj, ShardedTensorBase) or obj.is_cuda:
            return "<CUDA tensor>"

    s = repr(obj)
    if len(s) > PRINT_LOCAL_MAX_LENGTH:
        s = s[: PRINT_LOCAL_MAX_LENGTH // 2] + "..." + s[-PRINT_LOCAL_MAX_LENGTH // 2 :]
    return s


def get_rss_gb() -> float:
    """Return current process RSS in GB. Cheap: reads one line from /proc."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB → GB
    except Exception:
        pass
    return 0.0


def get_mem_available_gb() -> float:
    """Return available memory in GB, cgroup-aware.
    Checks cgroup limit first (container/k8s), falls back to /proc/meminfo."""
    try:
        # cgroup v1
        limit = int(open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read().strip())
        usage = int(open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read().strip())
        return (limit - usage) / (1024 ** 3)
    except Exception:
        pass
    try:
        # cgroup v2
        limit_s = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit_s == "max":
            raise ValueError("no cgroup limit")
        limit = int(limit_s)
        usage = int(open("/sys/fs/cgroup/memory.current").read().strip())
        return (limit - usage) / (1024 ** 3)
    except Exception:
        pass
    try:
        # fallback: host-wide
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB → GB
    except Exception:
        pass
    return float("inf")


if __name__ == "__main__":
    # Example usage

    def my_callback() -> None:
        print("Watchdog timed out in callback.")
        # Demonstrate an error
        raise ValueError("Example error from callback.")

    print("Simple usage example:")
    wd = Watchdog(timeout=2, callback=my_callback, enabled=True)
    print("Sleeping 3s so the watchdog times out.")
    time.sleep(30)
    # Because we never reset or finish, the watchdog should fire and
    # forcibly exit, after printing the traceback and stack dumps.
    print("You won't see this line if the watchdog fired first.")
