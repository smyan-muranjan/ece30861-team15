from typing import Any
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def calculate(self, repository_data: Any) -> float:
        pass
