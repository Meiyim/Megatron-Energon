# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""
Global handle managers for tar and idx files.

Provides process-wide LRU caches that replace the per-reader local caches in
ITarReader and CachedItarOffsetReader.  The managers are auto-created with
defaults on first access; custom limits can be passed via ``get_savable_loader``
(``max_tar_handles`` / ``max_idx_handles``).

The integration points live in the classes themselves:
- ``ITarReader._get_itarfile_cached``  uses ``GlobalTarHandleManager``
- ``ITarReader._get_item_by_sample_pointer``  retries with handle eviction
- ``CachedItarOffsetReader.get_itar_byte_offset``  uses ``GlobalIdxHandleManager``
- ``BlendDataset.save_state / restore_state``  path-based incremental resume
- ``PackingDataset._fill_reading_buffer``  catches SkipSample
"""

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.energon.flavors.webdataset.itar import ITarFile
    from megatron.energon.flavors.webdataset.itar_reader import ITarReader

logger = logging.getLogger(__name__)


# ============================================================================
# Global Tar Handle Manager
# ============================================================================

HandleKey = Tuple[int, int]


@dataclass
class HandleEntry:
    """Cache entry for an open ITarFile handle."""
    tar_file: "ITarFile"
    reader: "ITarReader"
    tar_file_id: int
    access_count: int = 0


class GlobalTarHandleManager:
    """
    Process-wide singleton LRU cache for ITarFile handles.

    When there are 500+ data sources each ITarReader independently manages its own
    handle cache, causing handle counts to accumulate and AFS cache invalidation to
    make training progressively slower.  This manager provides a single global LRU
    across all ITarReader instances.
    """

    _instance: Optional["GlobalTarHandleManager"] = None

    def __init__(self, max_handles: int = 100, log_interval: int = 5000):
        self.max_handles = max_handles
        self.log_interval = log_interval
        self.handles: OrderedDict[HandleKey, HandleEntry] = OrderedDict()
        self.total_accesses = 0
        self.total_evictions = 0
        self.total_opens = 0
        logger.info(f"[GlobalTarHandleManager] Initialized with max_handles={max_handles}")

    @classmethod
    def get_instance(cls, max_handles: int = 100, log_interval: int = 5000) -> "GlobalTarHandleManager":
        """Return the singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls(max_handles=max_handles, log_interval=log_interval)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            for entry in cls._instance.handles.values():
                try:
                    if entry.tar_file.fileobj:
                        entry.tar_file.fileobj.close()
                    entry.tar_file.close()
                except Exception:
                    pass
            cls._instance.handles.clear()
        cls._instance = None

    def acquire(self, reader: "ITarReader", tar_file_id: int) -> "ITarFile":
        """Return the cached ITarFile, opening and registering it if necessary."""
        key = (id(reader), tar_file_id)

        if key in self.handles:
            self.handles.move_to_end(key)
            self.handles[key].access_count += 1
            self.total_accesses += 1
            self._maybe_log()
            return self.handles[key].tar_file

        tar_file = self._open_tar_file(reader, tar_file_id)
        entry = HandleEntry(tar_file=tar_file, reader=reader, tar_file_id=tar_file_id, access_count=1)
        self.handles[key] = entry
        self.total_opens += 1
        self.total_accesses += 1
        self._evict_if_needed()
        self._maybe_log()
        return tar_file

    def _open_tar_file(self, reader: "ITarReader", tar_file_id: int) -> "ITarFile":
        from megatron.energon.flavors.webdataset.itar import ITarFile
        file_object = reader.tar_filepaths[tar_file_id].open(mode="rb", prefetch_file=True)
        return ITarFile.open(fileobj=file_object, mode="r:")

    def evict(self, reader_id: int, tar_file_id: int) -> bool:
        """Force-evict a handle (e.g. after a truncated/corrupt connection).

        Returns True if the handle was found and evicted.
        """
        key = (reader_id, tar_file_id)
        if key in self.handles:
            entry = self.handles.pop(key)
            self._close_handle(entry)
            logger.info(
                f"[GlobalTarHandleManager] Evicted bad handle: "
                f"reader_id={reader_id}, tar_file_id={tar_file_id}"
            )
            return True
        return False

    def _evict_if_needed(self):
        while len(self.handles) > self.max_handles:
            key, entry = self.handles.popitem(last=False)
            self._close_handle(entry)
            self.total_evictions += 1

    def _close_handle(self, entry: HandleEntry):
        try:
            if entry.tar_file.fileobj:
                entry.tar_file.fileobj.close()
            entry.tar_file.close()
        except Exception as e:
            logger.warning(f"[GlobalTarHandleManager] Error closing handle: {e}")
        if hasattr(entry.reader, "itar_files_cache") and entry.tar_file_id in entry.reader.itar_files_cache:
            del entry.reader.itar_files_cache[entry.tar_file_id]

    def _maybe_log(self):
        if self.total_accesses % self.log_interval == 0:
            hit_rate = (
                (self.total_accesses - self.total_opens) / self.total_accesses * 100
                if self.total_accesses > 0 else 0
            )
            logger.info(
                f"[GlobalTarHandleManager] pid={os.getpid()} "
                f"handles={len(self.handles)}/{self.max_handles}, "
                f"accesses={self.total_accesses}, "
                f"opens={self.total_opens}, "
                f"evictions={self.total_evictions}, "
                f"hit_rate={hit_rate:.1f}%"
            )

    def get_stats(self) -> dict:
        return {
            "current_handles": len(self.handles),
            "max_handles": self.max_handles,
            "total_accesses": self.total_accesses,
            "total_opens": self.total_opens,
            "total_evictions": self.total_evictions,
            "hit_rate": (
                (self.total_accesses - self.total_opens) / self.total_accesses * 100
                if self.total_accesses > 0 else 0
            ),
        }


# ============================================================================
# Global Idx Handle Manager
# ============================================================================

IdxHandleKey = str


@dataclass
class IdxHandleEntry:
    """Cache entry for an open TarIndexReader handle."""
    tar_index_reader: Any  # TarIndexReader
    tar_file_path: str
    access_count: int = 0


class GlobalIdxHandleManager:
    """
    Process-wide singleton LRU cache for TarIndexReader (.tar.idx) handles.

    Mirrors GlobalTarHandleManager but for index files.
    """

    _instance: Optional["GlobalIdxHandleManager"] = None

    def __init__(self, max_handles: int = 100, log_interval: int = 5000):
        self.max_handles = max_handles
        self.log_interval = log_interval
        self.handles: OrderedDict[IdxHandleKey, IdxHandleEntry] = OrderedDict()
        self.total_accesses = 0
        self.total_evictions = 0
        self.total_opens = 0
        logger.info(f"[GlobalIdxHandleManager] Initialized with max_handles={max_handles}")

    @classmethod
    def get_instance(cls, max_handles: int = 100, log_interval: int = 5000) -> "GlobalIdxHandleManager":
        """Return the singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls(max_handles=max_handles, log_interval=log_interval)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            for entry in cls._instance.handles.values():
                try:
                    entry.tar_index_reader.close()
                except Exception:
                    pass
            cls._instance.handles.clear()
        cls._instance = None

    def acquire(self, tar_file_path: str) -> Any:
        """Return a cached TarIndexReader, opening one if necessary."""
        from megatron.energon.flavors.webdataset.itar import TarIndexReader

        key = str(tar_file_path)
        if key in self.handles:
            self.handles.move_to_end(key)
            self.handles[key].access_count += 1
            self.total_accesses += 1
            self._maybe_log()
            return self.handles[key].tar_index_reader

        tar_index_reader = TarIndexReader(tar_file_path)
        entry = IdxHandleEntry(tar_index_reader=tar_index_reader, tar_file_path=key, access_count=1)
        self.handles[key] = entry
        self.total_opens += 1
        self.total_accesses += 1
        self._evict_if_needed()
        self._maybe_log()
        return tar_index_reader

    def _evict_if_needed(self):
        while len(self.handles) > self.max_handles:
            key, entry = self.handles.popitem(last=False)
            self._close_handle(entry)
            self.total_evictions += 1

    def _close_handle(self, entry: IdxHandleEntry):
        try:
            entry.tar_index_reader.close()
        except Exception as e:
            logger.warning(f"[GlobalIdxHandleManager] Error closing handle: {e}")

    def _maybe_log(self):
        if self.total_accesses % self.log_interval == 0:
            hit_rate = (
                (self.total_accesses - self.total_opens) / self.total_accesses * 100
                if self.total_accesses > 0 else 0
            )
            logger.info(
                f"[GlobalIdxHandleManager] pid={os.getpid()} "
                f"handles={len(self.handles)}/{self.max_handles}, "
                f"accesses={self.total_accesses}, "
                f"opens={self.total_opens}, "
                f"evictions={self.total_evictions}, "
                f"hit_rate={hit_rate:.1f}%"
            )

    def get_stats(self) -> dict:
        return {
            "current_handles": len(self.handles),
            "max_handles": self.max_handles,
            "total_accesses": self.total_accesses,
            "total_opens": self.total_opens,
            "total_evictions": self.total_evictions,
            "hit_rate": (
                (self.total_accesses - self.total_opens) / self.total_accesses * 100
                if self.total_accesses > 0 else 0
            ),
        }


# ============================================================================
# Helpers
# ============================================================================

def _get_dataset_path(ds) -> Optional[str]:
    """
    Recursively locate a dataset's path identifier, traversing wrapper layers.
    Used by BlendDataset incremental-resume to build path-to-state maps.
    """
    if hasattr(ds, "path"):
        return str(ds.path)
    if hasattr(ds, "join_readers") and ds.join_readers:
        reader = ds.join_readers[0]
        if hasattr(reader, "tar_filepaths") and reader.tar_filepaths:
            return os.path.dirname(str(reader.tar_filepaths[0]))
    if hasattr(ds, "datasets") and ds.datasets:
        for sub_ds in ds.datasets:
            path = _get_dataset_path(sub_ds)
            if path:
                return path
    if hasattr(ds, "dataset"):
        return _get_dataset_path(ds.dataset)
    return None


# ============================================================================
# S3 fork-safety patch (external library — must remain a runtime patch)
# ============================================================================

_s3_fork_safety_patched = False


def _patch_s3_fork_safety():
    """
    Patch S3StorageProvider (multistorageclient) for fork-safety.

    After fork(), child workers inherit boto3 / rust S3 clients that share a
    urllib3 connection pool with the parent.  Multiple workers reading/writing on
    the same sockets causes interleaved HTTP bytes and FlexibleChecksumError.

    This patch recreates both clients lazily on the first S3 operation after fork.
    """
    global _s3_fork_safety_patched
    if _s3_fork_safety_patched:
        return

    try:
        from multistorageclient.providers.s3 import S3StorageProvider
        from multistorageclient.types import RetryableError
    except ImportError:
        logger.warning("[S3_FORK_SAFETY] multistorageclient not available, skipping")
        return

    _orig_create = S3StorageProvider._create_s3_client

    def _patched_create(self, *args, **kwargs):
        client = _orig_create(self, *args, **kwargs)
        self._s3_create_args = args
        self._s3_create_kwargs = kwargs
        self._s3_client_pid = os.getpid()
        return client

    S3StorageProvider._create_s3_client = _patched_create

    _orig_create_rust = S3StorageProvider._create_rust_client

    def _patched_create_rust(self, rust_client_options=None):
        self._rust_client_options_saved = rust_client_options
        return None  # created lazily in workers

    S3StorageProvider._create_rust_client = _patched_create_rust

    _orig_translate = S3StorageProvider._translate_errors
    import threading
    _fork_safety_lock = threading.Lock()
    _main_process_pid = os.getpid()

    def _patched_translate(self, func, operation, bucket, key):
        current_pid = os.getpid()
        if current_pid != getattr(self, "_s3_fork_ready_pid", None):
            with _fork_safety_lock:
                if current_pid != getattr(self, "_s3_fork_ready_pid", None):
                    is_child = current_pid != _main_process_pid
                    if current_pid != getattr(self, "_s3_client_pid", None):
                        saved_args = getattr(self, "_s3_create_args", ())
                        saved_kwargs = getattr(self, "_s3_create_kwargs", {})
                        self._s3_client = self._create_s3_client(*saved_args, **saved_kwargs)
                        logger.info(
                            f"[S3_FORK_SAFETY] Recreated S3 client: "
                            f"pid={current_pid}, operation={operation}, bucket={bucket}"
                        )
                    if is_child and self._rust_client is None:
                        rust_opts = getattr(self, "_rust_client_options_saved", None)
                        if rust_opts is not None:
                            self._rust_client = _orig_create_rust(self, rust_opts)
                            logger.info(
                                f"[S3_FORK_SAFETY] Created fresh Rust client: "
                                f"pid={current_pid}, operation={operation}, bucket={bucket}"
                            )
                    self._s3_fork_ready_pid = current_pid

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                return _orig_translate(self, func, operation, bucket, key)
            except (RuntimeError, RetryableError) as e:
                err_msg = str(e)
                is_retryable = isinstance(e, RetryableError) or any(
                    kw in err_msg
                    for kw in (
                        "SlowDown", "429", "503", "RateLimitExceeded",
                        "RequestRateLimitExceeded", "Throttl", "TooManyRequest",
                        "Failed to GET", "Failed to PUT",
                    )
                )
                if not is_retryable or attempt >= max_retries:
                    raise
                wait = min(1.0 * (2 ** attempt), 30.0)
                logger.warning(
                    f"[S3_RATE_LIMIT] {operation} {bucket}/{key}: {err_msg}, "
                    f"retry {attempt + 1}/{max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)

    S3StorageProvider._translate_errors = _patched_translate

    _s3_fork_safety_patched = True
    logger.info("[S3_FORK_SAFETY] S3StorageProvider patched for fork-safety")
