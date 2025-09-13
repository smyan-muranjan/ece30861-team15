import unittest
from unittest.mock import MagicMock, patch

from src.api.HuggingFaceClient import HuggingFaceClient
from src.models.dataset_stats import DatasetStats


class TestGetDatasetInfo(unittest.TestCase):
    def setUp(self):
        self.client = HuggingFaceClient()
        self.mock_api = MagicMock()
        self.client.api = self.mock_api

    def test_valid_dataset_info(self):
        # Mock info object returned by api.dataset_info
        mock_info = MagicMock()
        mock_info.likes = 100
        mock_info.downloads = 1000
        self.mock_api.dataset_info.return_value = mock_info

        # Patch constants for normalization
        with patch('src.constants.MAX_DATASET_LIKES', 1000), \
             patch('src.constants.MAX_DATASET_DOWNLOADS', 10000):
            stats = self.client.get_dataset_info('namespace/dataset_name')
            self.assertIsInstance(stats, DatasetStats)
            self.assertGreaterEqual(stats.normalized_likes, 0)
            self.assertLessEqual(stats.normalized_likes, 1)
            self.assertGreaterEqual(stats.normalized_downloads, 0)
            self.assertLessEqual(stats.normalized_downloads, 1)

    def test_zero_likes_and_downloads(self):
        mock_info = MagicMock()
        mock_info.likes = 0
        mock_info.downloads = 0
        self.mock_api.dataset_info.return_value = mock_info
        with patch('src.constants.MAX_DATASET_LIKES', 1000), \
             patch('src.constants.MAX_DATASET_DOWNLOADS', 10000):
            stats = self.client.get_dataset_info('namespace/dataset_name')
            self.assertEqual(stats.normalized_likes, 0.0)
            self.assertEqual(stats.normalized_downloads, 0.0)

    def test_large_likes_and_downloads(self):
        mock_info = MagicMock()
        mock_info.likes = 1000000
        mock_info.downloads = 10000000
        self.mock_api.dataset_info.return_value = mock_info
        with patch('src.constants.MAX_DATASET_LIKES', 1000), \
             patch('src.constants.MAX_DATASET_DOWNLOADS', 10000):
            stats = self.client.get_dataset_info('namespace/dataset_name')
            self.assertEqual(stats.normalized_likes, 1.0)
            self.assertEqual(stats.normalized_downloads, 1.0)

    def test_negative_likes_and_downloads(self):
        mock_info = MagicMock()
        mock_info.likes = -5
        mock_info.downloads = -10
        self.mock_api.dataset_info.return_value = mock_info
        with patch('src.constants.MAX_DATASET_LIKES', 1000), \
             patch('src.constants.MAX_DATASET_DOWNLOADS', 10000):
            stats = self.client.get_dataset_info('namespace/dataset_name')
            self.assertEqual(stats.normalized_likes, 0.0)
            self.assertEqual(stats.normalized_downloads, 0.0)

    def test_api_exception(self):
        self.mock_api.dataset_info.side_effect = Exception('API error')
        with self.assertRaises(Exception):
            self.client.get_dataset_info('namespace/dataset_name')


if __name__ == '__main__':
    unittest.main()
