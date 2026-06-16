# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
from enum import Enum
from typing import List, Set

import braceexpand

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml, load_yaml_json
from megatron.energon.flavors.webdataset.config import (
    INDEX_SQLITE_FILENAME,
    INFO_JSON_FILENAME,
    INFO_YAML_FILENAME,
    MAIN_FOLDER_NAME,
)
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.structs import (
    ShardInfo,
    WebdatasetInfo,
    WebdatasetSplits,
)
from megatron.energon.metadata_cache import metadata_cache
from megatron.energon.typed_converter import JsonParser


class EnergonDatasetType(Enum):
    METADATASET = "metadataset"
    WEBDATASET = "webdataset"
    JSONL = "jsonl"
    FILESYSTEM = "filesystem"
    INVALID = "invalid"


@edataclass
class WebdatasetMeta:
    """Class for getting metadata from a webdataset."""

    sample_excludes: Set[str]
    shards: List[ShardInfo]
    split_part_files: List[str]
    info_shard_files: List[str]

    @staticmethod
    def from_config(
        path: EPath,
        *,
        split_part: str,
        split_config: str | None = None,
    ) -> "WebdatasetMeta":
        """
        Loads the metadata for a webdataset, i.e. the shards and sample excludes.

        Args:
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            split_config: Config file to use for shard split definitions.
        """
        if split_config is None:
            split_config = "split.yaml"

        parser = JsonParser(strict=True)
        info_object = get_dataset_info(path)

        info = parser.raw_to_typed(
            info_object,
            WebdatasetInfo,
        )
        try:
            splits = parser.raw_to_typed(
                load_yaml_json(path / MAIN_FOLDER_NAME / split_config),
                WebdatasetSplits,
            )
        except FileNotFoundError:
            if split_config == "split.yaml":
                # Try split.json instead
                splits = parser.raw_to_typed(
                    load_yaml_json(path / MAIN_FOLDER_NAME / "split.json"),
                    WebdatasetSplits,
                )
            else:
                raise
        assert split_part in splits.split_parts, f"Invalid split part: {split_part!r}"
        split_excludes = {
            excluded
            for excluded in splits.exclude
            for excluded in braceexpand.braceexpand(excluded)
        }

        all_split_part_files = [
            name
            for name in splits.split_parts[split_part]
            for name in braceexpand.braceexpand(name)
        ]

        split_part_files = [name for name in all_split_part_files if name not in split_excludes]
        if len(split_part_files) == 0:
            raise EmptyDatasetError(f"No shards found in split part {split_part!r}")
        return WebdatasetMeta(
            sample_excludes={excluded for excluded in split_excludes if "/" in excluded},
            shards=[
                ShardInfo(
                    name=name,
                    path=path / name,
                    count=info.shard_counts[name],
                )
                for name in split_part_files
            ],
            split_part_files=all_split_part_files,
            info_shard_files=list(info.shard_counts.keys()),
        )


def get_info_shard_files(path: EPath) -> List[str]:
    """Use this if you don't need the full metadata for split parts, but just the shard files."""
    parser = JsonParser(strict=True)
    info = parser.raw_to_typed(
        get_dataset_info(path),
        WebdatasetInfo,
    )
    return list(info.shard_counts.keys())


def get_dataset_info(path: EPath) -> dict:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return the dataset info as a dict.
    """
    cached, hit = metadata_cache.get("dataset_info", str(path))
    if hit:
        return cached

    info_config = path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME
    # YAML for backwards compatibility
    yaml_info_config = path / MAIN_FOLDER_NAME / ".info.yaml"

    if info_config.is_file():
        with info_config.open("r") as rf:
            result = json.load(rf)
    elif yaml_info_config.is_file():
        result = load_yaml(yaml_info_config.read_bytes())
    else:
        raise ValueError(f"No info config file found at {info_config} or {yaml_info_config}")

    metadata_cache.put("dataset_info", str(path), result)
    return result


def check_dataset_info_present(path: EPath) -> bool:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return True if the dataset info is present, False otherwise.
    """
    cached, hit = metadata_cache.get("info_present", str(path))
    if hit:
        return cached
    result = (path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME).is_file() or (
        path / MAIN_FOLDER_NAME / INFO_YAML_FILENAME
    ).is_file()
    metadata_cache.put("info_present", str(path), result)
    return result


def get_dataset_type(path: EPath) -> EnergonDatasetType:
    """Get the type of the dataset at the given path.

    Args:
        path: The path to the dataset as specified by the user.

    Returns:
        The type of the dataset.
    """
    cached, hit = metadata_cache.get("dataset_type", str(path))
    if hit:
        return cached

    metadata_db = path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME

    if path.is_file():
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            result = EnergonDatasetType.JSONL
        elif path.name.endswith(".yaml"):
            result = EnergonDatasetType.METADATASET
        elif _sniff_jsonl(path):
            # Extension-less file whose first non-empty line is a JSON object →
            # treat as JSONL. Robust against arbitrary part naming (part-00000).
            result = EnergonDatasetType.JSONL
        else:
            result = EnergonDatasetType.INVALID
    elif check_dataset_info_present(path):
        result = EnergonDatasetType.WEBDATASET
    elif metadata_db.is_file():
        # There is an sqlite, but no .info.json or .info.yaml,
        # so it's a filesystem dataset
        result = EnergonDatasetType.FILESYSTEM
    elif path.is_dir() and _dir_is_jsonl(path):
        # A directory of jsonl part files (no .nv-meta) is a single logical
        # JSONL dataset split across parts. Detected by extension OR by
        # sniffing the first part's first line as JSON (handles part-00000
        # with no extension).
        result = EnergonDatasetType.JSONL
    else:
        result = EnergonDatasetType.INVALID

    metadata_cache.put("dataset_type", str(path), result)
    return result


def _sniff_jsonl(path: EPath) -> bool:
    """Return True if the file's first non-empty line is a JSON object.

    Supports plain and gzip-compressed files. Used so JSONL detection does not
    depend solely on file extension (e.g. AFS exports named ``part-00000``).
    """
    try:
        name = path.name
        if name.endswith(".gz") or name.endswith(".jsonl.gz"):
            import gzip

            # GzipFile.close() does NOT close an externally-supplied fileobj, so
            # open the raw handle in its own context manager to avoid leaking it.
            with path.open("rb") as raw:
                with gzip.GzipFile(fileobj=raw) as f:
                    head = _first_nonblank_line(f)
        else:
            with path.open("rb") as f:
                head = _first_nonblank_line(f)
    except Exception:
        return False
    if head is None:
        return False
    head = head.strip()
    if not head.startswith(b"{") or not head.endswith(b"}"):
        return False
    import json as _json

    try:
        _json.loads(head)
        return True
    except Exception:
        return False


def _first_nonblank_line(f, *, max_lines: int = 64) -> "bytes | None":
    """Return the first line that is not blank (whitespace-only), or None.

    Bounded to ``max_lines`` so a file of blank lines can't spin forever.
    """
    for _ in range(max_lines):
        line = f.readline(1 << 16)
        if not line:
            return None
        if line.strip():
            return line
    return None


# Index sidecars that sit next to data parts and must be ignored when sniffing
# a directory for JSONL content.
_JSONL_SIDECAR_SUFFIXES = (".jsonl.idx", ".jsonl.idx.tmp", ".gzidx")


def _dir_is_jsonl(path: EPath) -> bool:
    """Return True if a directory looks like a (multi-part) JSONL dataset.

    First tries extension globs (fast); otherwise sniffs candidate child files'
    first line as JSON so extension-less part files are recognized too. Index
    sidecars (.jsonl.idx / .gzidx) are skipped, and several candidates are tried
    (not just the first) before giving up.
    """
    if next(path.glob("*.jsonl"), None) is not None:
        return True
    if next(path.glob("*.jsonl.gz"), None) is not None:
        return True
    candidates = sorted(
        (
            p
            for p in path.glob("*")
            if p.is_file()
            and not any(p.name.endswith(s) for s in _JSONL_SIDECAR_SUFFIXES)
        ),
        key=lambda p: p.name,
    )
    for c in candidates[:4]:
        if _sniff_jsonl(c):
            return True
    return False
