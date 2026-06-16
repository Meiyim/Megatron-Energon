# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Transparent gzip support for JSONL datasets.

energon's JSONL flavor indexes samples by *uncompressed* byte offset and uses
``seek()``/``read()`` for random access. Plain ``.jsonl`` files support this
natively. For ``.gz`` files we use ``indexed_gzip.IndexedGzipFile``, which
exposes seek/tell/read/readline in the *uncompressed* coordinate space — so the
exact same offset index (``.jsonl.idx``) works unchanged.

This lets cook read compressed ``.gz`` sources in place: no decompression to
disk (which would inflate ~3-4x and blow local storage on large pools).

A per-file gzip seek-point index (``<path>.gzidx``) is built during ``prepare``
and imported on read, so random seeks don't re-decompress from the start.
"""

from __future__ import annotations

import gzip
from typing import BinaryIO

from megatron.energon.epathlib import EPath

GZIDX_SUFFIX = ".gzidx"
# Seek point every 4 MiB of uncompressed data — trade index size vs seek cost.
_GZ_SPACING = 4 * 1024 * 1024


def is_gzip(path: EPath) -> bool:
    """True if the path looks like a gzip-compressed file."""
    return path.name.endswith(".gz")


def open_seekable(path: EPath, *, build_gzidx: bool = False) -> BinaryIO:
    """Open a jsonl source for seek/tell/read/readline in uncompressed space.

    Plain files: returns the normal binary handle.
    ``.gz`` files (local): returns an ``IndexedGzipFile`` whose seek/tell operate
    on uncompressed offsets. If a sidecar ``<path>.gzidx`` exists it is imported
    for fast random access; if ``build_gzidx`` is set and it is missing, a full
    index is built and exported (done once during prepare).
    """
    if not is_gzip(path):
        return path.open("rb")

    try:
        import indexed_gzip as igzip
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Reading gzip-compressed JSONL requires indexed_gzip. "
            "Install with: pip install indexed_gzip"
        ) from e

    if not path.is_local():
        # IndexedGzipFile needs a real local file for efficient random access.
        # Remote gz would have to be downloaded first; not supported here.
        raise NotImplementedError(
            f"Random access into remote gzip is not supported: {path}. "
            "Use a local path or an uncompressed .jsonl."
        )

    local = str(path.local_path())
    gzidx_local = local + GZIDX_SUFFIX

    import os as _os

    if _os.path.isfile(gzidx_local):
        f = igzip.IndexedGzipFile(local, index_file=gzidx_local)
    else:
        f = igzip.IndexedGzipFile(local, spacing=_GZ_SPACING)
        if build_gzidx:
            f.build_full_index()
            try:
                f.export_index(gzidx_local)
            except OSError:
                pass  # read-only dir etc.; seeking still works, just slower next time
    return f


def open_sequential(path: EPath) -> BinaryIO:
    """Open for a purely sequential pass (prepare's line scan).

    For ``.gz`` we slurp the whole compressed file into memory in one large
    read, then decompress from RAM. This matters on networked mounts (AFS/NFS):
    Python's ``gzip`` reads the underlying file in small chunks, and each small
    read pays a full network round-trip — which throttles a remote ``.gz`` to a
    crawl even though bulk bandwidth is high. One big read saturates bandwidth;
    decompression is then pure CPU.
    """
    if is_gzip(path):
        import gzip
        import io

        if hasattr(path, "read_bytes"):
            raw = path.read_bytes()
        else:
            with path.open("rb") as f:
                raw = f.read()
        return gzip.GzipFile(fileobj=io.BytesIO(raw))
    return path.open("rb")
