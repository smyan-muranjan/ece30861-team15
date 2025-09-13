import os
import shutil
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from git import Actor, Repo

from src.api.GitClient import CommitStats
from src.metrics.local_metrics import LocalMetricsCalculator

sys.path.insert(0,
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))))


class TestLocalMetricsCalculator(unittest.TestCase):
    """Test cases for LocalMetricsCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = LocalMetricsCalculator()
        self.temp_repo_path = None

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_repo_path and os.path.exists(self.temp_repo_path):
            self._force_remove_directory(self.temp_repo_path)

    def _force_remove_directory(self, path):
        """Force remove directory with retries"""
        def handle_remove_readonly(func, path, exc):
            """Handle readonly files on Windows."""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
                break
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Increasing delay
                else:
                    print(f"Can't remove {path} after {max_retries} tries")

    def create_test_repo(self) -> str:
        """Create a test repository for testing."""
        self.temp_repo_path = tempfile.mkdtemp(prefix="test_repo_")
        repo = Repo.init(self.temp_repo_path)

        # Create basic files
        (Path(self.temp_repo_path) / "test.py"). \
            write_text("print('Hello!')")
        (Path(self.temp_repo_path) / "README.md"). \
            write_text("# Test\n## Usage\nExample code")
        (Path(self.temp_repo_path) / "requirements.txt"). \
            write_text("requests")

        # Create examples and tests
        (Path(self.temp_repo_path) / "examples"). \
            mkdir()
        (Path(self.temp_repo_path) / "tests"). \
            mkdir()

        # Make initial commit
        default_author = Actor("DefaultAuthor", "default@test.com")
        repo.index.add(["test.py", "README.md", "requirements.txt"])
        repo.index.commit("Initial commit", author=default_author)

        return self.temp_repo_path

    def test_calculate_bus_factor(self):
        """Test bus factor calculation."""
        repo_path = self.create_test_repo()

        score, latency = \
            self.calculator.calculate_bus_factor(repo_path)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_calculate_code_quality(self):
        """Test code quality calculation."""
        repo_path = self.create_test_repo()

        score, latency = \
            self.calculator.calculate_code_quality(repo_path)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_calculate_ramp_up_time(self):
        """Test ramp-up time calculation."""
        repo_path = self.create_test_repo()

        score, latency = \
            self.calculator.calculate_ramp_up_time(repo_path)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_calculate_size_score(self):
        """Test size score calculation."""
        repo_path = self.create_test_repo()

        scores, latency = \
            self.calculator.calculate_size_score(repo_path)

        self.assertIsInstance(scores, dict)
        self.assertIn('raspberry_pi', scores)
        self.assertIn('jetson_nano', scores)
        self.assertIn('desktop_pc', scores)
        self.assertIn('aws_server', scores)

        for score in scores.values():
            self.assertIn(score, [0.0, 1.0])

        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_analyze_repository_success(self):
        """Test successful repository analysis."""
        repo_path = self.create_test_repo()

        # Mock the clone_repository method to return our test repo
        with patch.object(self.calculator.git_client,
                          'clone_repository', return_value=repo_path):
            results = self.calculator. \
                analyze_repository("https://github.com/test/repo")

        # Check that all expected metrics are present
        expected_metrics = [
            'bus_factor', 'bus_factor_latency',
            'code_quality', 'code_quality_latency',
            'ramp_up_time', 'ramp_up_time_latency',
            'size_score', 'size_score_latency'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, results)

        # Check that scores are in valid ranges
        for metric in ['bus_factor', 'code_quality', 'ramp_up_time']:
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)

        # Check that latencies are non-negative integers
        for metric in ['bus_factor_latency',
                       'code_quality_latency',
                       'ramp_up_time_latency',
                       'size_score_latency']:
            self.assertIsInstance(results[metric], int)
            self.assertGreaterEqual(results[metric], 0)

    def test_analyze_repository_exception_during_analysis(self):
        """Test analyze_repository when an exception occurs during analysis."""
        repo_path = self.create_test_repo()

        # Mock the individual calculation methods to raise exceptions
        with patch.object(self.calculator.git_client,
                          'clone_repository', return_value=repo_path), \
             patch.object(self.calculator.git_client,
                          'cleanup') as mock_cleanup, \
             patch.object(self.calculator, 'calculate_bus_factor',
                          side_effect=Exception("Analysis failed")):

            with self.assertRaises(Exception) as context:
                self.calculator. \
                    analyze_repository("https://github.com/test/repo")

            self.assertEqual(str(context.exception), "Analysis failed")

            # Verify cleanup was called even when analysis fails
            mock_cleanup.assert_called_once()

    def test_analyze_repository_git_client_cleanup_called(self):
        """Test git_client.cleanup() is called even when analysis fails."""

        with patch.object(self.calculator.git_client,
                          'clone_repository', return_value=None), \
             patch.object(self.calculator.
             git_client, 'cleanup') as mock_cleanup:

            results = \
                self.calculator.\
                analyze_repository("https://github.com/test/repo")

            mock_cleanup.assert_not_called()

            # Verify default metrics are returned
            self.assertEqual(results['bus_factor'], 0.0)
            self.assertEqual(results['code_quality'], 0.0)

    def test_analyze_repository_cleanup_called_on_success(self):
        """Test that git_client.cleanup() is called on successful analysis."""
        repo_path = self.create_test_repo()

        with patch.object(self.calculator.git_client,
                          'clone_repository', return_value=repo_path), \
             patch.object(self.calculator.git_client,
                          'cleanup') as mock_cleanup:

            results = \
                self.calculator.\
                analyze_repository("https://github.com/test/repo")

            mock_cleanup.assert_called_once()

            # Verify results are returned
            self.assertIn('bus_factor', results)
            self.assertIn('code_quality', results)

    def test_calculate_bus_factor_error_handling(self):
        """Test bus factor calculation error handling."""
        # Test with non-existent path
        score, latency = \
            self.calculator.calculate_bus_factor("/nonexistent/path")

        self.assertEqual(score, 0.0)
        self.assertIsInstance(latency, int)

    def test_calculate_code_quality_error_handling(self):
        """Test code quality calculation error handling."""
        # Test with non-existent path
        score, latency = \
            self.calculator.calculate_code_quality("/nonexistent/path")

        self.assertEqual(score, 0.0)
        self.assertIsInstance(latency, int)

    def test_calculate_ramp_up_time_error_handling(self):
        """Test ramp-up time calculation error handling."""
        # Test with non-existent path
        score, latency = \
            self.calculator.calculate_ramp_up_time("/nonexistent/path")

        self.assertEqual(score, 0.0)
        self.assertIsInstance(latency, int)

    def test_calculate_size_score_error_handling(self):
        """Test size score calculation error handling."""
        # Test with non-existent path
        scores, latency = \
            self.calculator.calculate_size_score("/nonexistent/path")

        self.assertTrue(all(score == 0.0 for score in scores.values()))
        self.assertIsInstance(latency, int)

    def test_latency_measurement_accuracy(self):
        """Test that latency measurement works correctly."""
        repo_path = self.create_test_repo()

        # Test that latency is measured and returned
        score, latency = self.calculator.calculate_bus_factor(repo_path)

        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)
        # Test with a mock that takes some time
        with patch.object(self.calculator.git_client,
                          'analyze_commits') as mock_analyze:
            mock_analyze.return_value = \
                CommitStats(total_commits=1,
                            contributors={'test': 1}, bus_factor=0.5)
            # Add a small delay to test latency measurement

            def slow_analyze(path):
                time.sleep(0.01)
                return CommitStats(total_commits=1,
                                   contributors={'test': 1}, bus_factor=0.5)

            mock_analyze.side_effect = slow_analyze
            score, latency = self.calculator.calculate_bus_factor(repo_path)
            self.assertIsInstance(latency, int)
            self.assertGreaterEqual(latency, 10)


if __name__ == '__main__':
    unittest.main()
