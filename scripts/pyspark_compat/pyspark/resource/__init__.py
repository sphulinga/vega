"""Resource profile stubs for upstream PySpark tests."""

from __future__ import annotations


class _ResourceRequest:
    def __init__(self, amount, discoveryScript=None, resourceName=None, vendor=None):
        self.amount = amount
        self.discoveryScript = discoveryScript
        self.resourceName = resourceName
        self.vendor = vendor


class ExecutorResourceRequests:
    def __init__(self):
        self.requests = {}

    def cores(self, amount: int):
        self.requests["cores"] = _ResourceRequest(amount)
        return self

    def memory(self, value: str):
        self.requests["memory"] = _ResourceRequest(_parse_memory(value))
        return self

    def memoryOverhead(self, value: str):
        self.requests["memoryOverhead"] = _ResourceRequest(_parse_memory(value))
        return self

    def pysparkMemory(self, value: str):
        self.requests["pyspark.memory"] = _ResourceRequest(_parse_memory(value))
        return self

    def resource(self, name: str, amount, discoveryScript=None, vendor=None):
        self.requests[name] = _ResourceRequest(
            amount,
            discoveryScript=discoveryScript,
            resourceName="gpu" if name == "gpu" else name,
            vendor=vendor,
        )
        return self


class TaskResourceRequests:
    def __init__(self):
        self.requests = {}

    def cpus(self, amount: float):
        self.requests["cpus"] = _ResourceRequest(amount)
        return self

    def resource(self, name: str, amount: float):
        self.requests[name] = _ResourceRequest(amount)
        return self


class ResourceProfile:
    def __init__(self, executorResources, taskResources):
        self.executorResources = executorResources
        self.taskResources = taskResources


class ResourceProfileBuilder:
    def __init__(self):
        self._executor = None
        self._task = None

    def require(self, reqs):
        if isinstance(reqs, ExecutorResourceRequests):
            self._executor = reqs.requests
        elif isinstance(reqs, TaskResourceRequests):
            self._task = reqs.requests
        return self

    @property
    def build(self):
        return ResourceProfile(self._executor or {}, self._task or {})


def _parse_memory(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("g"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("m"):
        return int(value[:-1])
    return int(value)
