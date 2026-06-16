# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from bisect import bisect_right
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
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.source_info import SourceInfo

T_index = TypeVar("T_index", covariant=False)


class IJsonlReader(ABC):
    """
    Class for reading one or more indexed jsonl files as a single sample stream.

    A source may be split into many part files (e.g. ``part-00000.jsonl.gz`` …).
    All parts are presented as one contiguous, globally-indexed dataset: global
    sample index ``i`` maps to ``(part_j, local_i)`` via the cumulative per-part
    sample counts (bisect), mirroring how ``ShardInfosITarReader`` joins tar
    shards. This keeps the number of energon *blend entries* equal to the number
    of *sources* (not parts), so metadataset init stays cheap even with thousands
    of parts per source.

    Args:
        jsonl_path: A single jsonl path, or a list of part paths in stream order.
        sample_filter: An optional filter function to select samples by their key.
        index_cache_size: The size of the per-part index cache.
    """

    jsonl_paths: List[EPath]
    sample_filter: Optional[Callable[[str], bool]]

    def __init__(
        self,
        jsonl_path: "EPath | str | List[EPath] | List[str]",
        sample_filter: Optional[Callable[[str], bool]] = None,
        index_cache_size: int = 5,
    ):
        if isinstance(jsonl_path, (list, tuple)):
            self.jsonl_paths = [EPath(p) for p in jsonl_path]
        else:
            self.jsonl_paths = [EPath(jsonl_path)]
        assert self.jsonl_paths, "IJsonlReader needs at least one jsonl path"

        self.sample_filter = sample_filter
        self._index_cache_size = index_cache_size

        # Per-part sample counts and cumulative offsets for global→(part,local).
        self._part_counts: List[int] = [
            IJsonlIndexReader.count_samples(p) for p in self.jsonl_paths
        ]
        self._cum: List[int] = [0]
        for c in self._part_counts:
            self._cum.append(self._cum[-1] + c)

        # Lazily-opened per-part readers/files.
        self._offset_readers: List[Optional[CachedIJsonlOffsetReader]] = [
            None
        ] * len(self.jsonl_paths)
        self._files: List[Optional[IJsonlFile]] = [None] * len(self.jsonl_paths)

    def __len__(self) -> int:
        return self._cum[-1]

    def __str__(self) -> str:
        return f"IJsonlReader(n_parts={len(self.jsonl_paths)}, first={self.jsonl_paths[0]})"

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        """Map a global sample index to (part_index, local_index)."""
        if global_idx < 0 or global_idx >= self._cum[-1]:
            raise IndexError(f"index {global_idx} out of range [0, {self._cum[-1]})")
        part = bisect_right(self._cum, global_idx) - 1
        return part, global_idx - self._cum[part]

    def _offset_reader(self, part: int) -> CachedIJsonlOffsetReader:
        r = self._offset_readers[part]
        if r is None:
            r = CachedIJsonlOffsetReader(
                self.jsonl_paths[part], cache_size=self._index_cache_size
            )
            self._offset_readers[part] = r
        return r

    def _file(self, part: int) -> IJsonlFile:
        from megatron.energon.flavors.jsonl.gzip_support import open_seekable

        f = self._files[part]
        if f is None:
            f = IJsonlFile(open_seekable(self.jsonl_paths[part]))
            self._files[part] = f
        return f

    def _get_item_by_sample_pointer(
        self,
        part: int,
        sample_pointer: IJsonlSamplePointer,
        global_idx: int,
    ) -> FilteredSample | None:
        """Get a sample from a specific part, or slice it."""
        key = str(global_idx)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None

        json_data = self._file(part).next(
            sample_pointer.byte_offset, sample_pointer.byte_size
        )
        if json_data is None:
            return None

        shard_name = self.jsonl_paths[part].name
        return FilteredSample(
            __key__=key,
            __shard__=shard_name,
            __restore_key__=("Webdataset", global_idx),
            __sources__=(
                SourceInfo(
                    dataset_path=str(self.jsonl_paths[part]),
                    index=sample_pointer.index,
                    shard_name=shard_name,
                    file_names=(f"{key}.json",),
                ),
            ),
            json=json_data,
        )

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        """
        Get a sample from the dataset by global index.
        """

        assert isinstance(idx, (int, str)), f"Invalid argument type for __getitem__: {type(idx)}"
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

        part, local_idx = self._locate(idx)
        byte_offset, byte_size = self._offset_reader(part).get_ijsonl_byte_offset(local_idx)
        sample: FilteredSample | None = self._get_item_by_sample_pointer(
            part,
            IJsonlSamplePointer(
                index=local_idx,
                byte_offset=byte_offset,
                byte_size=byte_size,
            ),
            global_idx=idx,
        )

        if sample is None:
            return None

        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample["json"], sample["__sources__"][0]
        else:
            return sample

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all samples across all parts.

        Returns:
            A generator of tuples of (sample_key, size, tar_file_id)
        """
        for part, path in enumerate(self.jsonl_paths):
            last_byte_offset = 0
            base = self._cum[part]
            with IJsonlIndexReader(path) as ijsonl_index_reader:
                for local_idx, byte_offset in enumerate(ijsonl_index_reader):
                    if last_byte_offset == byte_offset:
                        continue
                    yield str(base + local_idx), byte_offset - last_byte_offset, 0
                    last_byte_offset = byte_offset

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample parts across all parts.

        Returns:
            A generator of tuples of (sample_key + "." + part_name, size, tar_file_id)
        """
        for part, path in enumerate(self.jsonl_paths):
            last_byte_offset = 0
            base = self._cum[part]
            with IJsonlIndexReader(path) as ijsonl_index_reader:
                for local_idx, byte_offset in enumerate(ijsonl_index_reader):
                    if last_byte_offset == byte_offset:
                        continue
                    yield f"{base + local_idx}.json", byte_offset - last_byte_offset, 0
                    last_byte_offset = byte_offset

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        """Given a sample key, list all its parts. (E.g. given 1, list 1.jpg, 1.json, etc.)

        Args:
            sample_key: The sample key to list the parts of.

        Returns:
            A generator of tuples of (part_name, size, tar_file_id)
        """
        try:
            global_idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid JSONL sample key: {sample_key}")

        part, local_idx = self._locate(global_idx)
        _, byte_size = self._offset_reader(part).get_ijsonl_byte_offset(local_idx)
        yield f"{sample_key}.json", byte_size, 0

    def get_total_size(self) -> int:
        """Total uncompressed byte size summed across all parts."""
        total = 0
        for part in range(len(self.jsonl_paths)):
            total += self._offset_reader(part).get_total_size()
        return total

    def close(self):
        for f in self._files:
            if f is not None:
                f.close()
        for r in self._offset_readers:
            if r is not None:
                r.close()
        self._files = [None] * len(self.jsonl_paths)
        self._offset_readers = [None] * len(self.jsonl_paths)
