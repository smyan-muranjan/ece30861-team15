from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def calculate(self, repository_data: any) -> float:
        pass
