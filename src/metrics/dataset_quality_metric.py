from typing import Any

from src.metrics.metric import Metric
from src.models.dataset_stats import DatasetStats


class DatasetQualityMetric(Metric):
    LIKES_WEIGHT = 0.5
    DOWNLOADS_WEIGHT = 0.5

    def calculate(self, repository_data: Any) -> float:
        assert isinstance(repository_data, DatasetStats)
        return self.LIKES_WEIGHT * repository_data.normalized_likes + \
            self.DOWNLOADS_WEIGHT * repository_data.normalized_downloads
