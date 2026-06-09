"""PySpark-compatible API backed by PyVega."""

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.rdd import RDD
from pyspark.serializers import (
    BatchedSerializer,
    CloudPickleSerializer,
    CPickleSerializer,
    MarshalSerializer,
    NoOpSerializer,
    UTF8Deserializer,
)

__all__ = [
    "SparkConf",
    "SparkContext",
    "RDD",
    "CloudPickleSerializer",
    "CPickleSerializer",
    "BatchedSerializer",
    "MarshalSerializer",
    "UTF8Deserializer",
    "NoOpSerializer",
]
