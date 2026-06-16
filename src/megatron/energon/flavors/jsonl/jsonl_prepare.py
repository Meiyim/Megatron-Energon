# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.gzip_support import is_gzip, open_seekable, open_sequential
from megatron.energon.flavors.jsonl.ijsonl import IJsonlFile, IJsonlIndexWriter

logger = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)


class JsonlPreparator:
    @staticmethod
    def iter_dataset_content(
        path: Union[str, EPath],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield example dataset content for a few samples.

        Args:
            path: Path to the tar file.
        """
        with open_seekable(EPath(path)) as f:
            with IJsonlFile(f) as index_reader:
                for entry in index_reader:
                    yield {"json": entry}

    @classmethod
    def prepare_dataset(
        cls,
        path: Union[Path, EPath],
    ) -> int:
        """
        Preprocess the jsonl file. Preprocessing is done in parallel.
        Counts the number of samples.

        Args:
            path: Path to the jsonl file

        Returns:
            Count of samples in the jsonl file.
        """
        count = 0
        path = EPath(path)
        # Fast path: slurp the whole (decompressed) file once, then find line
        # boundaries with C-level scanning instead of a Python per-line
        # readline()/strip() loop (which tops out ~50 MB/s and dominates prepare
        # on large plain-JSONL parts). numpy locates all newlines at once and we
        # write the whole offset index in one buffer, so prepare is IO-bound.
        #
        # Index semantics (unchanged): byte offset (uncompressed stream) of the
        # start of every non-empty line, then a final sentinel == total size.
        import numpy as np

        # Stream the (decompressed) file in large blocks rather than one giant
        # read; on networked mounts block reads pipeline with kernel readahead.
        chunks = []
        with open_sequential(path) as f:
            while True:
                b = f.read(64 << 20)  # 64 MiB
                if not b:
                    break
                chunks.append(b)
        raw = b"".join(chunks) if len(chunks) != 1 else chunks[0]
        del chunks
        n = len(raw)
        assert n != 0, "File is empty."

        buf = np.frombuffer(raw, dtype=np.uint8)
        # start-of-line positions: 0, and every index right after a '\n'
        nl = np.flatnonzero(buf == 0x0A)
        starts = np.empty(len(nl) + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = nl + 1
        # drop a phantom start at EOF (file ending in '\n')
        if starts[-1] >= n:
            starts = starts[:-1]
        # Drop blank lines (those containing only whitespace up to the next
        # newline) but keep the ORIGINAL byte offset of every retained line —
        # including lines that begin with leading whitespace before the JSON.
        # ``ends`` is the next newline for each start (or EOF for the last line);
        # a line is blank iff stripping its bytes leaves nothing.
        if len(starts):
            ends = np.empty(len(starts), dtype=np.int64)
            ends[:-1] = starts[1:] - 1  # exclude the '\n' itself
            ends[-1] = n
            # Whitespace bytes: \t \n \v \f \r and space.
            ws = np.zeros(256, dtype=bool)
            for c in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20):
                ws[c] = True
            is_ws = ws[buf]
            # cumulative count of non-whitespace bytes, for O(1) range queries
            nonws_cumsum = np.concatenate(([0], np.cumsum(~is_ws)))
            nonws_in_line = nonws_cumsum[ends] - nonws_cumsum[starts]
            offsets = starts[nonws_in_line > 0]
        else:
            offsets = starts
        count = int(len(offsets))

        # write offsets + final sentinel (total size) as one little-endian u64 buffer
        out = np.empty(count + 1, dtype="<u8")
        out[:count] = offsets
        out[count] = n
        iw = IJsonlIndexWriter(path)
        try:
            iw.ijsonl.write(out.tobytes())
            iw.close(finalize=True)
        except BaseException:
            iw.close(finalize=False)
            raise

        # For gzip sources, also build & export the gzip seek-point index
        # (<path>.gzidx) now, so read-time random access doesn't re-decompress
        # from the start of the file on every seek. Plain files need no such
        # index (seek is native), so this is a no-op for them.
        if is_gzip(path):
            try:
                open_seekable(path, build_gzidx=True).close()
            except Exception as e:  # noqa: BLE001
                # gzidx is a read-time optimization; failing to build it (e.g.
                # read-only dir, missing indexed_gzip) must not fail prepare.
                logger.warning("Could not build gzip seek index for %s: %s", path, e)
        return count


