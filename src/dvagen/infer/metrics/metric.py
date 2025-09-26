from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Iterable


class BaseMetric(ABC):
    @abstractmethod
    def __init__(self, predictions: list[str], references: list[str] = None, **kwargs): ...

    @abstractmethod
    def compute(self) -> dict[str, float]: ...


class MetricList(UserList):
    def __init__(self, metrics: Iterable[BaseMetric] = None):
        super().__init__(metrics)

    def compute(self) -> dict[str, float]:
        results = {}
        for metric in self.data:
            if not isinstance(metric, BaseMetric):
                raise TypeError(f"Expected BaseMetric, got {type(metric)}")
            results.update(metric.compute())
        return results
