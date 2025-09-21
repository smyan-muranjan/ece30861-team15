from typing import Any, Dict

from src.metrics.metric import Metric


class SizeMetric(Metric):
    async def calculate(self, metric_input: Any) -> Dict[str, float]:
        assert isinstance(metric_input, str)
        return self.git_client.get_repository_size(metric_input)
