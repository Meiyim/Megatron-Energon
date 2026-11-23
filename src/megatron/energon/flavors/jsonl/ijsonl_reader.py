# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from bisect import bisect_right
from collections import OrderedDict
from typing import (
    Callable,
    Generator,
    List,
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
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo
from megatron.energon.source_info import SourceInfo

T_index = TypeVar("T_index", covariant=False)


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


class ShardInfosIJsonlReader:
    """Random-access reader over a list of jsonl shards, as one logical dataset.

    The jsonl analogue of :class:`ShardInfosITarReader`: a global sample index is
    demultiplexed across the shards via ``shard_count_cumsum`` + ``bisect`` into
    ``(shard, local_index)``, then read from that shard's file. This is what lets
    one source (a directory of jsonl parts) be a single dataset with many shards,
    exactly like a WebDataset over many tars — so blend weight stays at the source
    level and parts are just shards.

    Per-shard single-file :class:`IJsonlReader` instances are held in an LRU bounded
    by ``index_cache_size`` to cap open file handles; set it to the worker's
    ``parallel_shard_iters`` so every concurrently-active shard stays open.

    The sample's ``__key__`` / ``__restore_key__`` / ``SourceInfo.index`` use the
    GLOBAL index (matching :class:`ShardInfosITarReader`), so savable restore
    addresses the same sample; ``__shard__`` / ``shard_name`` name the specific part.
    """

    base_path: EPath
    shard_infos: List[ShardInfo]
    shard_count_cumsum: List[int]

    def __init__(
        self,
        base_path: EPath,
        shard_infos: List[ShardInfo],
        sample_filter: Optional[Callable[[str], bool]] = None,
        index_cache_size: int = 5,
    ):
        self.base_path = base_path
        self.shard_infos = shard_infos
        self.sample_filter = sample_filter
        self.index_cache_size = max(1, index_cache_size)

        self.shard_count_cumsum = [0]
        total = 0
        for shardinfo in shard_infos:
            total += shardinfo.count
            self.shard_count_cumsum.append(total)

        # LRU of per-shard single-file readers, keyed by shard index.
        self._readers: "OrderedDict[int, IJsonlReader]" = OrderedDict()

    def __len__(self) -> int:
        return self.shard_count_cumsum[-1]

    def __str__(self) -> str:
        return (
            f"ShardInfosIJsonlReader(len={len(self)}, base_path={self.base_path}, "
            f"n_shards={len(self.shard_infos)})"
        )

    def _reader_for(self, shard_idx: int) -> IJsonlReader:
        reader = self._readers.get(shard_idx)
        if reader is not None:
            self._readers.move_to_end(shard_idx)
            return reader
        if len(self._readers) >= self.index_cache_size:
            _, oldest = self._readers.popitem(last=False)
            oldest.close()
        reader = IJsonlReader(
            self.shard_infos[shard_idx].path,
            sample_filter=self.sample_filter,
            index_cache_size=1,
        )
        self._readers[shard_idx] = reader
        return reader

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        full_entry_name = False
        if isinstance(idx, str):
            num_idx = idx
            if idx.endswith(".json"):
                num_idx = idx.removesuffix(".json")
                full_entry_name = True
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid JSONL sample key: {idx}")

        shard_idx = bisect_right(self.shard_count_cumsum, idx) - 1
        if shard_idx < 0 or shard_idx >= len(self.shard_infos):
            raise IndexError(f"Index out of bounds: {idx}")
        local_idx = idx - self.shard_count_cumsum[shard_idx]

        reader = self._reader_for(shard_idx)
        byte_offset, byte_size = reader.cached_offset_reader.get_ijsonl_byte_offset(local_idx)
        # index=idx keeps the GLOBAL index in the key/restore/SourceInfo, while the
        # per-shard reader stamps __shard__ / shard_name with this part's file name.
        sample = reader._get_item_by_sample_pointer(
            IJsonlSamplePointer(index=idx, byte_offset=byte_offset, byte_size=byte_size)
        )
        if sample is None:
            return None
        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample["json"], sample["__sources__"][0]
        return sample

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
