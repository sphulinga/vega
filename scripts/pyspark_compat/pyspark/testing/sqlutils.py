"""SQL testing utilities stub for upstream PySpark tests."""

from __future__ import annotations

import os

from pyspark.testing.utils import SPARK_HOME

__all__ = ["SPARK_HOME", "ReusedSQLTestCase", "SQLTestUtils"]


class SQLTestUtils:
    pass


class ReusedSQLTestCase:
    pass
