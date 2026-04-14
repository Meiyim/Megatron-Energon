# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.deprecation import warn_deprecated
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.metadata import check_dataset_info_present
from megatron.energon.flavors.webdataset.structs import DatasetSubset
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDatasetList,
)
from megatron.energon.worker import WorkerConfig


@edataclass
class DatasetReference:
    path: Union[str, EPath]
    split_part: Optional[str] = None
    # Note: subflavor is only for legacy compatibility.
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: Optional[str] = None
    split_config: Optional[str] = None

    weight: float = 1.0

    _dataset: Optional[DatasetLoaderInterface] = None

    def __post_init__(self):
        if self.subflavor is not None:
            warn_deprecated(
                "subflavor is deprecated, use subflavors instead. This will be removed in a future release."
            )
            if self.subflavors is None:
                self.subflavors = {"__subflavor__": self.subflavor}
            elif "__subflavor__" not in self.subflavors:
                self.subflavors = {"__subflavor__": self.subflavor, **(self.subflavors or {})}
            self.subflavor = None

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        if self.path.is_file():
            assert self.dataset_config is None, "Must not set dataset_config"
            assert self.split_config is None, "Must not set split_config"
            self._dataset = load_config(
                self.path,
                default_type=Metadataset,
                default_kwargs=dict(path=self.path),
            )
            self._dataset.post_initialize()
        elif check_dataset_info_present(self.path):
            self._dataset = DatasetLoader(
                path=self.path,
                split_config=self.split_config,
                dataset_config=self.dataset_config,
            )
            self._dataset.post_initialize()
        else:
            raise FileNotFoundError(self.path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        assert self._dataset is not None

        if shuffle_over_epochs_multiplier is None or self.shuffle_over_epochs_multiplier is None:
            # If no shuffling is requested, this has override priority.
            new_shuffle_over_epochs_multiplier = None
        elif shuffle_over_epochs_multiplier == -1 or self.shuffle_over_epochs_multiplier == -1:
            # Next priority is sampling without replacement.
            new_shuffle_over_epochs_multiplier = -1
        else:
            # Otherwise, multiply the shuffle over epochs multiplier.
            new_shuffle_over_epochs_multiplier = (
                shuffle_over_epochs_multiplier * self.shuffle_over_epochs_multiplier
            )

        return self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=new_shuffle_over_epochs_multiplier,
            subset=subset,
            **kwargs,
        )


_logger = logging.getLogger(__name__)

_ENERGON_IO_WORKERS = int(os.environ.get("ENERGON_IO_WORKERS", 0))


@edataclass
class MetadatasetBlender:
    """Internal blending of the dataset."""

    datasets: List[DatasetReference]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        n_workers = _ENERGON_IO_WORKERS
        if n_workers > 1 and len(self.datasets) > 1:
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=min(n_workers, len(self.datasets))) as pool:
                futs = {pool.submit(ds.post_initialize, mds_path): ds for ds in self.datasets}
                for fut in as_completed(futs):
                    fut.result()  # propagate exceptions
            _logger.info(
                f"[ENERGON] MetadatasetBlender.post_initialize: "
                f"{len(self.datasets)} datasets in {time.monotonic() - t0:.2f}s "
                f"(threaded, workers={min(n_workers, len(self.datasets))})"
            )
        else:
            t0 = time.monotonic()
            for dataset in self.datasets:
                dataset.post_initialize(mds_path)
            _logger.info(
                f"[ENERGON] MetadatasetBlender.post_initialize: "
                f"{len(self.datasets)} datasets in {time.monotonic() - t0:.2f}s (sequential)"
            )

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        sum_weight = sum(dataset.weight for dataset in self.datasets)

        def _load_one(dataset):
            return dataset, dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                subset=subset,
                **kwargs,
            )

        n_workers = _ENERGON_IO_WORKERS
        if n_workers > 1 and len(self.datasets) > 1:
            t0 = time.monotonic()
            results = []
            with ThreadPoolExecutor(max_workers=min(n_workers, len(self.datasets))) as pool:
                futs = [pool.submit(_load_one, ds) for ds in self.datasets]
                # preserve order to keep deterministic blending
                results = [f.result() for f in futs]
            _logger.info(
                f"[ENERGON] MetadatasetBlender.get_datasets: "
                f"{len(self.datasets)} datasets in {time.monotonic() - t0:.2f}s "
                f"(threaded, workers={min(n_workers, len(self.datasets))})"
            )
        else:
            t0 = time.monotonic()
            results = [_load_one(ds) for ds in self.datasets]
            _logger.info(
                f"[ENERGON] MetadatasetBlender.get_datasets: "
                f"{len(self.datasets)} datasets in {time.monotonic() - t0:.2f}s (sequential)"
            )

        datasets = []
        for dataset, inner_result in results:
            if inner_result.blend_mode not in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.DATASET_WEIGHT,
            ):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset in inner_result.datasets:
                if inner_result.blend_mode == DatasetBlendMode.DATASET_WEIGHT:
                    assert isinstance(loaded_dataset.weight, float)
                else:
                    assert loaded_dataset.weight is None
                    loaded_dataset.weight = 1.0
                loaded_dataset.weight = loaded_dataset.weight * dataset.weight / sum_weight
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.DATASET_WEIGHT,
            datasets=datasets,
        )


class Metadataset(DatasetLoaderInterface):
    """Main entry for metadataset."""

    _path: EPath
    _splits: Dict[str, MetadatasetBlender]

    def __init__(
        self,
        path: Union[EPath, str],
        splits: Dict[str, MetadatasetBlender],
    ):
        """Create the metadataset"""
        self._path = EPath(path)
        self._splits = splits

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is None
        for split in self._splits.values():
            split.post_initialize(self._path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        return self._splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            subset=subset,
            **kwargs,
        )
