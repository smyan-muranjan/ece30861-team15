from typing import Any

from src.metric_inputs.dataset_stats import DatasetStats
from src.metrics.metric import Metric


class DatasetQualityMetric(Metric):
    LIKES_WEIGHT = 0.5
    DOWNLOADS_WEIGHT = 0.5

    async def calculate(self, metric_input: Any) -> float:
        assert isinstance(metric_input, DatasetStats)
        return self.LIKES_WEIGHT * metric_input.normalized_likes + \
            self.DOWNLOADS_WEIGHT * metric_input.normalized_downloads
