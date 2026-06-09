"""Minimal py4j protocol stubs for upstream PySpark tests."""


class Py4JJavaError(Exception):
    def __init__(self, msg, java_exception=None):
        super().__init__(msg)
        self.java_exception = java_exception
