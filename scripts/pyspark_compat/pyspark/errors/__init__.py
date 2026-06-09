"""PySpark error stubs."""

from __future__ import annotations


class PySparkException(Exception):
    pass


class Py4JJavaError(Exception):
    def __init__(self, msg, java_exception=None):
        super().__init__(msg)
        self.java_exception = java_exception
