# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

import torch

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import WorkerRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")

logger = logging.getLogger(__name__)


class BlendDataset(BaseWrapperDataset[T_sample, T_sample]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weighting.
    The datasets may be infinite. This dataset is always infinite.
    """

    datasets: List[SavableDataset[T_sample]]
    weights: Tuple[float, ...]
    dataset_weights: Sequence[Tuple[SavableDataset[T_sample], float]]
    exhausted: List[bool]
    _worker_rng: WorkerRng

    _savable_fields = ("exhausted", "_worker_rng")

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_sample], float],
        worker_config: WorkerConfig,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                between 0 and 1. The output samples are sampled from the input datasets with the
                given probabilities.
            worker_config: Configuration for the workers.
        """
        # datasets = [dataset for dataset, _weight in dataset_weights]
        self.datasets, self.weights = zip(*dataset_weights)

        super().__init__(self.datasets, worker_config=worker_config)

        self.dataset_weights = dataset_weights
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self.exhausted = [False] * len(self.weights)

    def len_worker(self, worker_idx: int | None = None) -> int:
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"

        # Create a list of datasets and their weights, but
        # set the weight to 0 if the dataset has no samples on this worker.

        dataset_iters = []
        weights = []
        for idx, (dataset, weight) in enumerate(self.dataset_weights):
            assert weight > 0, "All blending weights must be > 0"

            if dataset.worker_has_samples():
                dataset_iters.append(iter(dataset))
                weights.append(weight)
            else:
                dataset_iters.append(None)
                weights.append(0)
                self.exhausted[idx] = True

        weights = torch.tensor(weights, dtype=torch.float32)
        if weights.sum() == 0:
            raise RuntimeError(
                "There is a worker with no samples in any of the blended datasets. "
                "This can happen if you have a lot of workers and your dataset is too small. "
                "Currently this case is not supported."
            )

        # Some may already be exhausted on this worker when restoring a state.
        for idx, exhausted in enumerate(self.exhausted):
            if exhausted:
                weights[idx] = 0
                dataset_iters[idx] = None

        while True:
            ds_idx = self._worker_rng.choice_idx(probs=weights)

            if dataset_iters[ds_idx] is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iters[ds_idx])
            except StopIteration:
                dataset_iters[ds_idx] = None
                weights[ds_idx] = 0
                self.exhausted[ds_idx] = True
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
            else:
                yield add_sample_restore_key(sample, ds_idx, src=self)

        self.exhausted = [False] * len(self.dataset_weights)

    def save_state(self) -> FlexState:
        """Save state with dataset path information for incremental resume."""
        from megatron.energon.global_handle_manager import _get_dataset_path

        state = super().save_state()
        state["_dataset_paths"] = [_get_dataset_path(ds) for ds in self.datasets]
        return state

    def restore_state(self, state: FlexState) -> None:
        """
        Restore state using path-based matching for incremental resume.

        If the checkpoint contains ``_dataset_paths``, sub-datasets are matched
        by path rather than by index, so data sources can be added or removed
        between runs without breaking resume.  Falls back to index-based
        matching for old checkpoints (no path info).
        """
        from megatron.energon.global_handle_manager import _get_dataset_path

        saved_paths: Optional[list] = state.get("_dataset_paths")
        saved_states = state["datasets"]

        if saved_paths is None:
            # Old checkpoint without path info — fall back to index-based matching.
            # Requires old sources to keep their order; new sources appended at end.
            logger.warning(
                "[BlendDataset] No _dataset_paths in checkpoint (old format), "
                f"falling back to index-based matching "
                f"(saved={len(saved_states)}, current={len(self.datasets)})"
            )
            saved_paths = [f"__index_{i}__" for i in range(len(saved_states))]
            current_paths = [f"__index_{i}__" for i in range(len(self.datasets))]
        else:
            current_paths = [_get_dataset_path(ds) for ds in self.datasets]

        path_to_state = {p: s for p, s in zip(saved_paths, saved_states) if p}

        matched_count = new_count = 0
        matched_paths: list = []
        new_paths: list = []

        for ds, current_path in zip(self.datasets, current_paths):
            if current_path and current_path in path_to_state:
                ds.restore_state(path_to_state[current_path])
                matched_count += 1
                matched_paths.append(current_path)
            else:
                new_count += 1
                new_paths.append(current_path)

        removed_paths = set(p for p in saved_paths if p) - set(p for p in current_paths if p)
        removed_count = len(removed_paths)

        # Restore own state
        super(BaseWrapperDataset, self).restore_state(state)

        # Restore exhausted flags by path
        saved_exhausted = state.get("exhausted", [False] * len(saved_states))
        path_to_exhausted = {p: e for p, e in zip(saved_paths, saved_exhausted) if p}
        self.exhausted = [path_to_exhausted.get(p, False) for p in current_paths]

        def _fmt(paths, n=5):
            shown = list(paths)[:n]
            s = "[" + ", ".join(str(p) for p in shown) + "]"
            return s + (f" ... +{len(paths) - n} more" if len(paths) > n else "")

        msg = (
            f"[BlendDataset] Incremental Resume: "
            f"ckpt={len(saved_states)}, current={len(self.datasets)}, "
            f"matched={matched_count}, new={new_count}, removed={removed_count}"
        )
        if matched_count:
            msg += f", matched_datasets={_fmt(matched_paths)}"
        if removed_count:
            msg += f", removed={_fmt(removed_paths)}"
        if new_count:
            msg += f", new_datasets={_fmt(new_paths)}"
        logger.info(msg)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset_weights": [
                (dataset.config(), weight) for dataset, weight in self.dataset_weights
            ],
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"BlendDataset(dataset_weights={self.dataset_weights})"
