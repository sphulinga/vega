"""Testing helpers for running upstream PySpark tests against PyVega."""

from __future__ import annotations

import os
import unittest

from pyspark.context import SparkContext


SPARK_HOME = os.environ.get(
    "SPARK_HOME",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "spark")
    ),
)


def have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def have_pandas() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


def have_numpy() -> bool:
    try:
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


class PySparkTestCase(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext()

    def tearDown(self):
        self.sc.stop()


ReusedPySparkTestCase = PySparkTestCase


class QuietTest:
    def __init__(self, sc: SparkContext):
        self.sc = sc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
