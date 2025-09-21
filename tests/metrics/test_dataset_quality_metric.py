import pytest

from src.metric_inputs.dataset_stats import DatasetStats
from src.metrics.dataset_quality_metric import DatasetQualityMetric


class TestDatasetQualityMetric:
    def setup_method(self):
        self.metric = DatasetQualityMetric()

    @pytest.mark.asyncio
    async def test_calculate_typical(self):
        stats = DatasetStats(normalized_likes=0.8, normalized_downloads=0.6)
        result = await self.metric.calculate(stats)
        expected = 0.5 * 0.8 + 0.5 * 0.6
        assert abs(result - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_calculate_zero(self):
        stats = DatasetStats(normalized_likes=0.0, normalized_downloads=0.0)
        result = await self.metric.calculate(stats)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_one(self):
        stats = DatasetStats(normalized_likes=1.0, normalized_downloads=1.0)
        result = await self.metric.calculate(stats)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_calculate_invalid_type(self):
        with pytest.raises(AssertionError):
            await self.metric.calculate(
                {"normalized_likes": 0.5, "normalized_downloads": 0.5}
                )
