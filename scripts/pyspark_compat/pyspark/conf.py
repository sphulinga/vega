"""Minimal SparkConf stub for upstream PySpark tests."""

from __future__ import annotations

from typing import Any


class SparkConf:
    def __init__(self, loadDefaults: bool = True, _jconf=None):
        del loadDefaults, _jconf
        self._settings: dict[str, str] = {}

    def set(self, key: str, value: Any) -> "SparkConf":
        self._settings[key] = str(value)
        return self

    def setAppName(self, name: str) -> "SparkConf":
        return self.set("spark.app.name", name)

    def setMaster(self, master: str) -> "SparkConf":
        return self.set("spark.master", master)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._settings.get(key, default)

    def getAll(self) -> list[tuple[str, str]]:
        return sorted(self._settings.items())

    def contains(self, key: str) -> bool:
        return key in self._settings
