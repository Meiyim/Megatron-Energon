# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""
Opt-in metadata cache for deduplicating S3 reads across DP ranks.

During dataset construction Energon probes and reads small metadata files
(.nv-meta/.info.json, dataset.yaml, split.yaml, …) via several functions in
``metadata.py``, ``dataset_config.py`` and ``eyaml.py``.  For a blend of N
datasets every DP rank issues ~7N identical S3 calls.

This module provides a single shared cache.  Usage from the datamodule::

    metadata_cache.enable()              # start recording
    dataset = get_train_dataset(...)     # real S3 reads happen here
    cache = metadata_cache.export()      # {namespace: {key: value}} — picklable
    metadata_cache.disable()

    # broadcast *cache* to other ranks …

    metadata_cache.enable(cache)         # replay
    dataset = get_train_dataset(...)     # zero S3 metadata reads
    metadata_cache.disable()

Each cached function adds ~3 lines; all cached values are plain Python objects
(bool / dict / enum) so the broadcast is cheap and fully picklable.
"""

from typing import Any, Dict, Optional, Tuple


class MetadataCache:
    """Simple namespace→key→value cache with enable / disable lifecycle."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Optional[Dict[str, Dict[str, Any]]] = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._data is not None

    def enable(self, data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._data = data if data is not None else {}

    def disable(self) -> None:
        self._data = None

    def export(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self._data

    # -- get / put -----------------------------------------------------------

    def get(self, namespace: str, key: str) -> Tuple[Any, bool]:
        """Return ``(value, True)`` on hit, ``(None, False)`` on miss."""
        if self._data is None:
            return None, False
        ns = self._data.get(namespace)
        if ns is not None and key in ns:
            return ns[key], True
        return None, False

    def put(self, namespace: str, key: str, value: Any) -> None:
        if self._data is None:
            return
        self._data.setdefault(namespace, {})[key] = value


# Module-level singleton — imported by metadata.py / eyaml.py / dataset_config.py
metadata_cache = MetadataCache()
