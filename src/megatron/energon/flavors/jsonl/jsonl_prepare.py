# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import tempfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
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
        src = EPath(path)
        # Remote jsonl files (BOS/S3) opened via multistorageclient may return a
        # RemoteFileReader when the file exceeds the memory load limit (~512MB).
        # RemoteFileReader raises io.UnsupportedOperation for readline(), which
        # breaks jsonl indexing. Copy remote files to a local temp file first so
        # that readline()/tell()/seek() work natively.
        if not src.is_local():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                src.copy(EPath(tmp_path))
                with EPath(tmp_path).open("rb") as f:
                    with IJsonlFile(f) as index_reader:
                        for entry in index_reader:
                            yield {"json": entry}
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            with src.open("rb") as f:
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
        src = EPath(path)
        # Remote jsonl files (BOS/S3) opened via multistorageclient may return a
        # RemoteFileReader when the file exceeds the memory load limit (~512MB).
        # RemoteFileReader raises io.UnsupportedOperation for readline(), which
        # breaks jsonl indexing. Copy remote files to a local temp file first so
        # that readline()/tell()/seek() work natively. The index (.jsonl.idx) is
        # still written to the original remote path via IJsonlIndexWriter.
        tmp_path: Optional[str] = None
        if not src.is_local():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name
            logger.info(f"Copying remote jsonl to local temp file: {src.url} -> {tmp_path}")
            src.copy(EPath(tmp_path))
            read_path = EPath(tmp_path)
        else:
            read_path = src

        try:
            count = 0
            # Processing is lagging behind. The offsets include empty lines. The whole file must be covered!
            last_offset = 0
            with IJsonlIndexWriter(src) as iw:
                with read_path.open("rb") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            if last_offset:
                                last_offset = f.tell()
                            continue
                        assert line.startswith(b"{") and line.endswith(b"}"), (
                            f"Line {line} does not start and end with a json object {{}}."
                        )
                        iw.append(last_offset)
                        last_offset = f.tell()
                        count += 1
                    assert last_offset == f.tell(), (
                        f"The last offset {last_offset} does not match the file size {f.tell()}."
                    )
                    assert last_offset != 0, "File is empty."
                    iw.append(last_offset)
            return count
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
