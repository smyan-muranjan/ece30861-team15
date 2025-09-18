import unittest

from src.metrics.dataset_quality_metric import DatasetQualityMetric
from src.models.dataset_stats import DatasetStats


class TestDatasetQualityMetric(unittest.TestCase):
    def setUp(self):
        self.metric = DatasetQualityMetric()

    def test_calculate_typical(self):
        stats = DatasetStats(normalized_likes=0.8, normalized_downloads=0.6)
        result = self.metric.calculate(stats)
        expected = 0.5 * 0.8 + 0.5 * 0.6
        self.assertAlmostEqual(result, expected)

    def test_calculate_zero(self):
        stats = DatasetStats(normalized_likes=0.0, normalized_downloads=0.0)
        result = self.metric.calculate(stats)
        self.assertEqual(result, 0.0)

    def test_calculate_one(self):
        stats = DatasetStats(normalized_likes=1.0, normalized_downloads=1.0)
        result = self.metric.calculate(stats)
        self.assertEqual(result, 1.0)

    def test_calculate_invalid_type(self):
        with self.assertRaises(AssertionError):
            self.metric.calculate(
                {"normalized_likes": 0.5, "normalized_downloads": 0.5}
                )


if __name__ == "__main__":
    unittest.main()
