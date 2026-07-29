# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import fcntl
import hashlib
import logging
import os
import tempfile
import threading
import time
from abc import ABC
from typing import (
    Callable,
    Generator,
    Optional,
    Tuple,
    TypeVar,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.ijsonl import (
    CachedIJsonlOffsetReader,
    IJsonlFile,
    IJsonlIndexReader,
    IJsonlSamplePointer,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.source_info import SourceInfo

logger = logging.getLogger(__name__)

T_index = TypeVar("T_index", covariant=False)

# Size threshold: remote jsonl files larger than this are downloaded to a local
# temp file before reading, to avoid per-sample BOS seek overhead.
_LARGE_JSONL_THRESHOLD = 512 * 1024 * 1024  # 512 MB

# Chunk size for streaming download (avoids holding the whole file in memory
# during copy, but keeps chunks large enough to amortize per-request overhead).
_DOWNLOAD_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB

# Retry config for the download step.
_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_BACKOFF_BASE = 2.0  # seconds; actual sleep = base * 2^attempt

# In-process cache so threads within the same worker process don't re-download.
_LOCAL_JSONL_CACHE: dict[str, str] = {}
_LOCAL_JSONL_CACHE_LOCK = threading.Lock()


def _local_cache_path(remote_path: EPath) -> str:
    """Return a stable, node-local path for caching *remote_path*.

    The filename is deterministic (hash of URL) so all DataLoader worker
    processes on the same node converge on the same local file and the
    file-lock protocol below prevents redundant downloads.
    """
    url_hash = hashlib.sha256(str(remote_path.url).encode()).hexdigest()[:16]
    suffix = remote_path.name  # e.g. "data.jsonl"
    cache_dir = os.environ.get("ENERGON_CACHE_DIR", tempfile.gettempdir())
    return os.path.join(cache_dir, f"energon_jsonl_{url_hash}_{suffix}")


def _download_with_retry(remote_path: EPath, tmp_path: str) -> None:
    """Stream *remote_path* to *tmp_path* in chunks, with retry on failure.

    Uses chunked reads instead of EPath.copy so that:
    1. We never load the whole file into memory at once.
    2. Transient network errors retry from scratch (tmp file is truncated).
    """
    file_size = remote_path.size()
    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        try:
            with remote_path.open("rb") as src, open(tmp_path, "wb") as dst:
                downloaded = 0
                while True:
                    chunk = src.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
                    downloaded += len(chunk)
            if downloaded != file_size:
                raise IOError(
                    f"Download incomplete: got {downloaded} bytes, expected {file_size}"
                )
            return
        except Exception as exc:
            # Truncate the partial tmp file before retry.
            try:
                open(tmp_path, "wb").close()
            except OSError:
                pass
            if attempt < _DOWNLOAD_MAX_RETRIES - 1:
                sleep = _DOWNLOAD_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Download attempt %d/%d failed (%s), retrying in %.0fs: %s",
                    attempt + 1, _DOWNLOAD_MAX_RETRIES, exc, sleep, remote_path,
                )
                time.sleep(sleep)
            else:
                raise IOError(
                    f"Failed to download {remote_path} after {_DOWNLOAD_MAX_RETRIES} attempts"
                ) from exc


def _ensure_local_jsonl(remote_path: EPath) -> EPath:
    """Download *remote_path* to a stable local path (once per node) and return
    an EPath pointing at the local copy.

    Uses an exclusive file lock so that among all DataLoader worker processes on
    the same node only one actually downloads; the rest wait and reuse the file.
    Safe for both multiprocessing (fork/spawn) and multithreading.
    """
    local_path = _local_cache_path(remote_path)

    # Fast path: check in-process cache first (avoids lock overhead for threads
    # within the same worker process after the first call).
    with _LOCAL_JSONL_CACHE_LOCK:
        if local_path in _LOCAL_JSONL_CACHE and os.path.exists(local_path):
            return EPath(local_path)

    lock_path = local_path + ".lock"
    with open(lock_path, "w") as lock_fh:
        # LOCK_EX: blocks until we are the only holder — covers concurrent
        # DataLoader worker processes on the same node.
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            # Another process may have finished the download while we waited.
            if os.path.exists(local_path):
                logger.debug("Reusing cached jsonl: %s", local_path)
            else:
                tmp_path = local_path + ".tmp"
                logger.info(
                    "Downloading large remote jsonl (%.0f MB) to local cache: %s",
                    remote_path.size() / 1024**2,
                    local_path,
                )
                _download_with_retry(remote_path, tmp_path)
                os.replace(tmp_path, local_path)  # atomic on POSIX
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    with _LOCAL_JSONL_CACHE_LOCK:
        _LOCAL_JSONL_CACHE[local_path] = local_path
    return EPath(local_path)


class IJsonlReader(ABC):
    """
    Class for reading indexed jsonl files containing json samples.

    The common usage patterns and random-access interfaces are provided here.

    Args:
        base_path: The path to the dataset.
        jsonl_path: The path to the jsonl file.
        jsonl_filename: The jsonl file name.
        sample_filter: An optional filter function to select samples by their key.
        index_cache_size: The size of the index cache.
    """

    jsonl_path: EPath
    sample_filter: Optional[Callable[[str], bool]]

    cached_offset_reader: CachedIJsonlOffsetReader
    ijsonl_file: IJsonlFile | None = None

    def __init__(
        self,
        jsonl_path: EPath,
        sample_filter: Optional[Callable[[str], bool]] = None,
        index_cache_size: int = 5,
    ):
        # For large remote files, download once and read locally to avoid
        # per-sample BOS seek overhead.
        if not jsonl_path.is_local() and jsonl_path.size() > _LARGE_JSONL_THRESHOLD:
            jsonl_path = _ensure_local_jsonl(jsonl_path)
        self.jsonl_path = jsonl_path
        self.sample_filter = sample_filter
        self.cached_offset_reader = CachedIJsonlOffsetReader(
            jsonl_path, cache_size=index_cache_size
        )

    def __len__(self) -> int:
        return len(self.cached_offset_reader)

    def __str__(self) -> str:
        return f"IJsonlReader(jsonl_path={self.jsonl_path})"

    def _get_item_by_sample_pointer(
        self,
        sample_pointer: IJsonlSamplePointer,
    ) -> FilteredSample | None:
        """
        Get a sample from the dataset or slice it.

        Args:
            sample_pointer: The sample pointer to get the sample from.
            sample_index: The global index of the sample in the dataset.

        Returns:
            The sample or None if the sample is invalid.
        """

        key = str(sample_pointer.index)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None

        if self.ijsonl_file is None:
            self.ijsonl_file = IJsonlFile(self.jsonl_path.open("rb"))

        json_data = self.ijsonl_file.next(sample_pointer.byte_offset, sample_pointer.byte_size)
        if json_data is None:
            return None

        return FilteredSample(
            __key__=key,
            __shard__=self.jsonl_path.name,
            __restore_key__=("Webdataset", sample_pointer.index),
            __sources__=(
                SourceInfo(
                    dataset_path=str(self.jsonl_path),
                    index=sample_pointer.index,
                    shard_name=self.jsonl_path.name,
                    file_names=(f"{key}.json",),
                ),
            ),
            json=json_data,
        )

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        """
        Get a sample from the dataset.
        """

        assert isinstance(idx, (int, str)), f"Invalid argument type for __getitem__: {type(idx)}"
        full_entry_name = False
        if isinstance(idx, str):
            if idx.endswith(".json"):
                num_idx = idx.removesuffix(".json")
                full_entry_name = True
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid JSONL sample key: {idx}")

        byte_offset, byte_size = self.cached_offset_reader.get_ijsonl_byte_offset(idx)
        sample: FilteredSample | None = self._get_item_by_sample_pointer(
            IJsonlSamplePointer(
                index=idx,
                byte_offset=byte_offset,
                byte_size=byte_size,
            )
        )

        if sample is None:
            return None

        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample["json"], sample["__sources__"][0]
        else:
            return sample

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all samples in the jsonl file.

        Returns:
            A generator of tuples of (sample_key, size, tar_file_id)
        """
        last_byte_offset = 0
        with IJsonlIndexReader(self.jsonl_path) as ijsonl_index_reader:
            for sample_idx, byte_offset in enumerate(ijsonl_index_reader):
                if last_byte_offset == byte_offset:
                    continue
                yield str(sample_idx), byte_offset - last_byte_offset, 0
                last_byte_offset = byte_offset

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample parts in the jsonl file.

        Returns:
            A generator of tuples of (sample_key + "." + part_name, size, tar_file_id)
        """
        last_byte_offset = 0
        with IJsonlIndexReader(self.jsonl_path) as ijsonl_index_reader:
            for sample_idx, byte_offset in enumerate(ijsonl_index_reader):
                if last_byte_offset == byte_offset:
                    continue
                yield f"{sample_idx}.json", byte_offset - last_byte_offset, 0
                last_byte_offset = byte_offset

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        """Given a sample key, list all its parts. (E.g. given 1, list 1.jpg, 1.json, etc.)

        Args:
            sample_key: The sample key to list the parts of.

        Returns:
            A generator of tuples of (part_name, size, tar_file_id)
        """
        try:
            sample_idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid JSONL sample key: {sample_key}")

        _, byte_size = self.cached_offset_reader.get_ijsonl_byte_offset(sample_idx)
        yield f"{sample_key}.json", byte_size, 0

    def get_total_size(self) -> int:
        return self.cached_offset_reader.get_total_size()

    def close(self):
        if self.ijsonl_file is not None:
            self.ijsonl_file.close()
        self.cached_offset_reader.close()
