"""Local filesystem helpers for PyVega RDD I/O."""

from __future__ import annotations

import glob
import os
import pickle
import shutil
from typing import Any


def save_as_text_file(partitions: list[list[Any]], path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path, exist_ok=True)
    for index, part in enumerate(partitions):
        filename = os.path.join(path, f"part-{index:05d}")
        with open(filename, "wb") as handle:
            for item in part:
                if isinstance(item, bytes):
                    handle.write(item + b"\n")
                elif isinstance(item, str):
                    handle.write(item.encode("utf-8") + b"\n")
                else:
                    handle.write(str(item).encode("utf-8") + b"\n")


def save_as_pickle_file(partitions: list[list[Any]], path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path, exist_ok=True)
    for index, part in enumerate(partitions):
        filename = os.path.join(path, f"part-{index:05d}")
        with open(filename, "wb") as handle:
            pickle.dump(list(part), handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_text_file(path: str) -> list[str]:
    lines: list[str] = []
    for filename in sorted(glob.glob(os.path.join(path, "part-*"))):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                lines.append(line.rstrip("\n"))
    return lines


def read_pickle_file(path: str) -> list[Any]:
    data: list[Any] = []
    for filename in sorted(glob.glob(os.path.join(path, "part-*"))):
        with open(filename, "rb") as handle:
            data.extend(pickle.load(handle))
    return data
