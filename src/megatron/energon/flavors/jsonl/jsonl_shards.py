# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate the ``*.jsonl`` shards under a directory as one logical dataset.

Shared by ``get_dataset_type`` (type detection), ``CrudeJsonlDatasetFactory``
(building the shard list) and ernie-dataflow's prepare step, so all three agree
on which files a jsonl directory contains and in what order.
"""

from typing import Generator, Mapping

from megatron.energon.epathlib import EPath


def item_path(item):
    """The path of an inline-list source item, dropping any display tag.

    A list item is either a bare path (str/EPath) or a ``{"path": ..., "tag": ...}``
    mapping; the ``tag`` is display-only provenance and never affects which shards
    the source resolves to. (A 2-element ``[path, tag]`` tuple is NOT accepted: it
    is ambiguous with a two-path list.)
    """
    return item["path"] if isinstance(item, Mapping) else item


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


def resolve_jsonl_shards(path) -> list[EPath]:
    """Flatten a jsonl source spec into its ordered shard list.

    Accepts a single ``.jsonl`` file (-> itself), a directory (-> its ``*.jsonl``
    shards), or an explicit list mixing dirs and files (-> each resolved and
    concatenated in list order). The list form lets one logical source span files
    that do not share a directory, while the global cumsum still treats them as one
    shuffle pool. Within a directory entry the order is relative-path-sorted (stable
    across runs); the list order itself is honored so cumsum is reproducible.
    """
    items = list(path) if isinstance(path, (list, tuple)) else [path]
    out: list[EPath] = []
    seen: set[str] = set()
    for it in items:
        it = item_path(it)
        p = it if isinstance(it, EPath) else EPath(str(it))
        shards = [p] if p.name.endswith(".jsonl") else list_jsonl_shards(p)
        for s in shards:
            # Dedup (first-occurrence order): overlapping items (a dir plus a file
            # under it, or a dir listed twice) must not read the same shard twice,
            # which would inflate the cumsum and disagree with EDF's leaf counting.
            if s.url not in seen:
                seen.add(s.url)
                out.append(s)
    return out


def dir_has_jsonl(path: EPath) -> bool:
    """True if ``path`` is a directory holding at least one ``*.jsonl`` file."""
    try:
        return next(iter_jsonl_shards(path), None) is not None
    except OSError:
        return False
