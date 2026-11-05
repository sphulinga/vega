"""Filesystem helpers for PyVega RDD I/O.

Reading and writing runs *inside the server* (as tasks), like Spark executors:
``textFile`` plans byte-range splits and each task reads its own split, and
``saveAsTextFile`` writes one ``part-NNNNN`` file per partition in parallel.
For a local server this is the same filesystem as the client.
"""

from __future__ import annotations

import bz2
import glob
import gzip
import io
import lzma
import os
import pickle
import shutil
from typing import Any, Iterable, Iterator

_COMPRESSED = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}

# Hadoop codec class names (and short names) accepted by saveAsTextFile.
_CODECS = {
    "gzip": (".gz", gzip.open),
    "gzipcodec": (".gz", gzip.open),
    "org.apache.hadoop.io.compress.gzipcodec": (".gz", gzip.open),
    "bzip2": (".bz2", bz2.open),
    "bzip2codec": (".bz2", bz2.open),
    "org.apache.hadoop.io.compress.bzip2codec": (".bz2", bz2.open),
    "xz": (".xz", lzma.open),
    "lzma": (".xz", lzma.open),
}


def _visible(name: str) -> bool:
    base = os.path.basename(name)
    return not (base.startswith("_") or base.startswith("."))


def list_input_files(path: str) -> list[str]:
    """Expand a Spark-style input spec (comma list, globs, directories)."""
    files: list[str] = []
    for spec in str(path).split(","):
        spec = spec.strip()
        if not spec:
            continue
        if spec.startswith("file://"):
            spec = spec[len("file://") :]
        matches = sorted(glob.glob(spec)) if glob.has_magic(spec) else [spec]
        if not matches:
            raise FileNotFoundError(f"Input path does not exist: {spec}")
        for match in matches:
            if os.path.isdir(match):
                for root, dirs, names in os.walk(match):
                    dirs[:] = sorted(d for d in dirs if _visible(d))
                    files.extend(
                        os.path.join(root, n) for n in sorted(names) if _visible(n)
                    )
            elif os.path.exists(match):
                files.append(match)
            else:
                raise FileNotFoundError(f"Input path does not exist: {match}")
    return files


def plan_text_splits(path: str, min_partitions: int) -> list[tuple[str, int, int]]:
    """Split input files into ``(file, start, end)`` byte ranges.

    Uncompressed files are split into roughly ``total / min_partitions``-sized
    ranges; compressed files are not splittable (end = -1).
    """
    files = list_input_files(path)
    sizes = {f: os.path.getsize(f) for f in files}
    total = sum(sizes[f] for f in files if os.path.splitext(f)[1] not in _COMPRESSED)
    goal = max(1, -(-total // max(1, min_partitions)))  # ceil division
    splits: list[tuple[str, int, int]] = []
    for f in files:
        if os.path.splitext(f)[1] in _COMPRESSED:
            splits.append((f, 0, -1))
            continue
        size = sizes[f]
        if size == 0:
            splits.append((f, 0, 0))
            continue
        start = 0
        while start < size:
            end = min(size, start + goal)
            # Avoid a tiny trailing split (Hadoop's 10% slop).
            if size - end < goal * 0.1:
                end = size
            splits.append((f, start, end))
            start = end
    return splits


def _decode(line: bytes, use_unicode: bool):
    line = line.rstrip(b"\n")
    if line.endswith(b"\r"):
        line = line[:-1]
    return line.decode("utf-8") if use_unicode else line


def read_text_split(split: tuple[str, int, int], use_unicode: bool = True) -> Iterator[Any]:
    """Yield the lines of one split using Hadoop LineRecordReader rules.

    A split owns every line that *starts* inside ``[start, end)``; the first
    partial line is skipped unless the split begins at offset 0.
    """
    path, start, end = split
    ext = os.path.splitext(path)[1]
    if end < 0:
        with _COMPRESSED[ext](path, "rb") as handle:
            for line in handle:
                yield _decode(line, use_unicode)
        return
    if start >= end:
        return
    with open(path, "rb") as handle:
        handle.seek(start)
        pos = start
        if start != 0:
            handle.seek(start - 1)
            pos = start - 1
            pos += len(handle.readline())  # finish the line that started earlier
        while pos < end:
            line = handle.readline()
            if not line:
                break
            pos += len(line)
            yield _decode(line, use_unicode)


def read_whole_file(path: str, use_unicode: bool = True) -> tuple[str, Any]:
    ext = os.path.splitext(path)[1]
    opener = _COMPRESSED.get(ext, open)
    with opener(path, "rb") as handle:
        data = handle.read()
    return ("file:" + os.path.abspath(path), data.decode("utf-8") if use_unicode else data)


def read_binary_file(path: str) -> tuple[str, bytes]:
    with open(path, "rb") as handle:
        return ("file:" + os.path.abspath(path), handle.read())


def read_pickle_part(path: str) -> list[Any]:
    with open(path, "rb") as handle:
        out: list[Any] = []
        while True:
            try:
                chunk = pickle.load(handle)
            except EOFError:
                break
            out.extend(chunk)
        return out


def prepare_output_dir(path: str, overwrite: bool = True) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Output directory {path} already exists")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path, exist_ok=True)


def mark_success(path: str) -> None:
    with open(os.path.join(path, "_SUCCESS"), "wb"):
        pass


def _to_bytes(item: Any) -> bytes:
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return str(item).encode("utf-8")


def text_writer(path: str, codec: str | None):
    """Return a mapPartitionsWithIndex function that writes text part files."""
    suffix, opener = "", open
    if codec:
        key = codec.lower()
        if key not in _CODECS:
            raise ValueError(f"Unsupported compression codec: {codec}")
        suffix, opener = _CODECS[key]

    def write(index: int, iterator: Iterable[Any]) -> Iterator[int]:
        filename = os.path.join(path, f"part-{index:05d}{suffix}")
        count = 0
        with opener(filename, "wb") as raw:
            handle = io.BufferedWriter(raw) if opener is open else raw
            for item in iterator:
                handle.write(_to_bytes(item) + b"\n")
                count += 1
            if handle is not raw:
                handle.flush()
        yield count

    return write


def pickle_writer(path: str, batch_size: int = 1024):
    def write(index: int, iterator: Iterable[Any]) -> Iterator[int]:
        filename = os.path.join(path, f"part-{index:05d}")
        count = 0
        batch: list[Any] = []
        with open(filename, "wb") as handle:
            for item in iterator:
                batch.append(item)
                if len(batch) >= batch_size:
                    pickle.dump(batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    count += len(batch)
                    batch = []
            if batch or count == 0:
                pickle.dump(batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
                count += len(batch)
        yield count

    return write


def jsonl_writer(path: str):
    import json

    def write(index: int, iterator: Iterable[Any]) -> Iterator[int]:
        filename = os.path.join(path, f"part-{index:05d}.jsonl")
        count = 0
        with open(filename, "w", encoding="utf-8") as handle:
            for item in iterator:
                handle.write(json.dumps(item, default=str))
                handle.write("\n")
                count += 1
        yield count

    return write


# --- Backwards-compatible client-side helpers ------------------------------


def save_as_text_file(partitions: list[list[Any]], path: str) -> None:
    prepare_output_dir(path)
    writer = text_writer(path, None)
    for index, part in enumerate(partitions):
        list(writer(index, part))


def save_as_pickle_file(partitions: list[list[Any]], path: str) -> None:
    prepare_output_dir(path)
    writer = pickle_writer(path)
    for index, part in enumerate(partitions):
        list(writer(index, part))


def read_text_file(path: str) -> list[str]:
    lines: list[str] = []
    for split in plan_text_splits(path, 1):
        lines.extend(read_text_split(split))
    return lines


def read_pickle_file(path: str) -> list[Any]:
    data: list[Any] = []
    for f in list_input_files(path):
        data.extend(read_pickle_part(f))
    return data
