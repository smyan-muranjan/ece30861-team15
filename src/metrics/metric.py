from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    @abstractmethod
    def calculate(self, repository_data: Any) -> float:
        pass
