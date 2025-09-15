from src.metrics.metric import Metric
from src.models.dataset_stats import DatasetStats


class DatasetQualityMetric(Metric):
    def calculate(self, repository_data: any) -> float:
        assert isinstance(repository_data, DatasetStats)
        return 0.5 * repository_data.normalized_likes + \
            0.5 * repository_data.normalized_downloads
