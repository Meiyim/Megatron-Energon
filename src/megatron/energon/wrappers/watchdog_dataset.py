# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.watchdog import Watchdog, get_rss_gb, get_mem_available_gb
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")

# How often to log memory (in samples). 0 = disable.
MEM_LOG_EVERY_N = int(os.environ.get("ENERGON_MEM_LOG_EVERY_N", "1"))


class WatchdogDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wraps another dataset and watches the time it takes to yield samples."""

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        worker_config: WorkerConfig,
        timeout_seconds: Optional[float] = 60,
        initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
    ):
        super().__init__(dataset, worker_config=worker_config)
        self.timeout_seconds = timeout_seconds
        self.initial_timeout_seconds = initial_timeout_seconds
        self.fail_on_timeout = fail_on_timeout
        self._sample_count = 0

    def reset_state_own(self) -> None:
        pass

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def _watchdog_trigger(self) -> None:
        if self.fail_on_timeout:
            raise TimeoutError(
                f"Watchdog triggered. Sample processing took longer than {self.timeout_seconds} seconds."
            )
        else:
            warnings.warn(
                f"Watchdog triggered. Sample processing took longer than {self.timeout_seconds} seconds.",
                RuntimeWarning,
            )

    def _log_memory(self) -> None:
        """Print one-line memory status every N samples."""
        self._sample_count += 1
        if MEM_LOG_EVERY_N <= 0 or self._sample_count % MEM_LOG_EVERY_N != 0:
            return
        rss = get_rss_gb()
        avail = get_mem_available_gb()
        print(
            f"[MEM] pid={os.getpid()} sample={self._sample_count} "
            f"rss={rss:.1f}G avail={avail:.1f}G",
            flush=True,
        )

    def __iter__(self) -> Iterator[T_sample]:
        if self.timeout_seconds is None:
            for item in self.dataset:
                self._log_memory()
                yield item
        else:
            watchdog = Watchdog(
                timeout=self.timeout_seconds,
                initial_timeout=self.initial_timeout_seconds,
                callback=self._watchdog_trigger,
                enabled=False,
            )
            try:
                watchdog.enable()
                for item in self.dataset:
                    watchdog.disable()
                    self._log_memory()
                    yield item
                    watchdog.enable()
            finally:
                watchdog.disable()

    def config(self) -> Dict[str, Any]:
        return self.dataset.config()

    def __str__(self):
        return f"WatchdogDataset(dataset={self.dataset})"
