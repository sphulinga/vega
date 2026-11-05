"""Storage levels for RDD persistence.

Vega keeps persisted partitions in server memory; every level is accepted for
API compatibility and recorded on the RDD, and all are served from memory.
"""

from __future__ import annotations


class StorageLevel:
    def __init__(
        self,
        useDisk: bool,
        useMemory: bool,
        useOffHeap: bool,
        deserialized: bool,
        replication: int = 1,
    ):
        self.useDisk = useDisk
        self.useMemory = useMemory
        self.useOffHeap = useOffHeap
        self.deserialized = deserialized
        self.replication = replication

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StorageLevel) and (
            self.useDisk,
            self.useMemory,
            self.useOffHeap,
            self.deserialized,
            self.replication,
        ) == (
            other.useDisk,
            other.useMemory,
            other.useOffHeap,
            other.deserialized,
            other.replication,
        )

    def __hash__(self) -> int:
        return hash(
            (self.useDisk, self.useMemory, self.useOffHeap, self.deserialized, self.replication)
        )

    def __repr__(self) -> str:
        return "StorageLevel(%s, %s, %s, %s, %s)" % (
            self.useDisk,
            self.useMemory,
            self.useOffHeap,
            self.deserialized,
            self.replication,
        )

    def __str__(self) -> str:
        result = ""
        result += "Disk " if self.useDisk else ""
        result += "Memory " if self.useMemory else ""
        result += "OffHeap " if self.useOffHeap else ""
        result += "Deserialized " if self.deserialized else "Serialized "
        result += "%sx Replicated" % self.replication
        return result


StorageLevel.NONE = StorageLevel(False, False, False, False)
StorageLevel.DISK_ONLY = StorageLevel(True, False, False, False)
StorageLevel.DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
StorageLevel.DISK_ONLY_3 = StorageLevel(True, False, False, False, 3)
StorageLevel.MEMORY_ONLY = StorageLevel(False, True, False, False)
StorageLevel.MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
StorageLevel.MEMORY_AND_DISK = StorageLevel(True, True, False, False)
StorageLevel.MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
StorageLevel.OFF_HEAP = StorageLevel(True, True, True, False, 1)
StorageLevel.MEMORY_AND_DISK_DESER = StorageLevel(True, True, False, True)
