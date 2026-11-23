# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate the ``*.jsonl`` shards under a directory as one logical dataset.

Shared by ``get_dataset_type`` (type detection), ``CrudeJsonlDatasetFactory``
(building the shard list) and ernie-dataflow's prepare step, so all three agree
on which files a jsonl directory contains and in what order.
"""

from typing import Generator

from megatron.energon.epathlib import EPath


def _is_jsonl(path: EPath) -> bool:
    # ``*.jsonl`` only — never ``*.jsonl.idx`` (the sidecar index).
    return path.name.endswith(".jsonl")


def iter_jsonl_shards(path: EPath) -> Generator[EPath, None, None]:
    """Yield every ``*.jsonl`` file under ``path`` (recursively), unordered."""
    for p in path.glob("**/*.jsonl"):
        if _is_jsonl(p):
            yield p


def list_jsonl_shards(path: EPath) -> list[EPath]:
    """Sorted ``*.jsonl`` shards under a directory, one logical source's parts.

    Ordered by path RELATIVE TO ``path`` (not basename): two subdirectories can
    hold a same-named part (``a/part-00000.jsonl`` vs ``b/part-00000.jsonl``);
    sorting on basename alone leaves their order to the filesystem/glob, which is
    not guaranteed stable across runs — and that order fixes the global cumsum
    index, so an unstable order breaks cook reproducibility and savable restore.
    """
    return sorted(iter_jsonl_shards(path), key=lambda p: p.relative_to(path))


def dir_has_jsonl(path: EPath) -> bool:
    """True if ``path`` is a directory holding at least one ``*.jsonl`` file."""
    try:
        return next(iter_jsonl_shards(path), None) is not None
    except OSError:
        return False
