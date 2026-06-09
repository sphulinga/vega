"""Serializer stubs required by upstream PySpark tests."""

from __future__ import annotations

try:
    import cloudpickle as pickle_mod
except ImportError:  # pragma: no cover
    import pickle as pickle_mod


class CloudPickleSerializer:
    def dumps(self, obj):
        return pickle_mod.dumps(obj)

    def loads(self, data):
        return pickle_mod.loads(data)


CPickleSerializer = CloudPickleSerializer


class BatchedSerializer:
    def __init__(self, serializer, batchSize=1000):
        self.serializer = serializer
        self.batchSize = batchSize


class MarshalSerializer:
    pass


class UTF8Deserializer:
    pass


class NoOpSerializer:
    pass


def read_int(stream):
    data = stream.read(4)
    if len(data) < 4:
        raise EOFError
    return int.from_bytes(data, byteorder="big", signed=True)
