"""Minimal SparkSession stub for upstream PySpark RDD tests."""

from __future__ import annotations

from pyspark.context import SparkContext


class SparkSession:
    _active: "SparkSession | None" = None

    def __init__(self, sparkContext: SparkContext | None = None):
        self.sparkContext = sparkContext or SparkContext.getOrCreate()

    @classmethod
    def builder(cls):
        return _Builder()


class _Builder:
    def master(self, master: str):
        return self

    def appName(self, name: str):
        return self

    def config(self, key: str, value: str):
        return self

    def getOrCreate(self) -> SparkSession:
        if SparkSession._active is None:
            SparkSession._active = SparkSession()
        return SparkSession._active
