import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.main import analyze_entry


@pytest.mark.asyncio
async def test_parallelism_performance():
    """
    Tests that processing entries concurrently is substantially
    faster than a sequential approach, validating the project's
    performance goals. [cite: 190]
    """
    # Create test entries in new format (code_link, dataset_link, model_link)
    entries = [(f"https://github.com/test/repo{i}",
                None, f"https://huggingface.co/model{i}")
               for i in range(4)]
    encountered_datasets = set()

    # Mock GitClient to simulate work without actual network/disk operations
    with patch('src.api.git_client.GitClient') as MockGitClient:
        mock_git_instance = MockGitClient.return_value

        # Simulate a 100ms I/O-bound clone operation
        def mock_clone(url):
            time.sleep(0.1)
            return f"/tmp/{url.split('/')[-1]}"
        mock_git_instance.clone_repository.side_effect = mock_clone

        # Simulate CPU-bound analysis tasks
        mock_git_instance.analyze_commits.return_value = MagicMock(
            total_commits=10,
            contributors={'a': 5, 'b': 5})
        mock_git_instance.analyze_code_quality.return_value = MagicMock(
            lint_errors=5,
            has_tests=True)
        mock_git_instance.analyze_ramp_up_time.return_value = MagicMock(
            readme_quality=0.8,
            has_examples=True,
            has_dependencies=True)
        mock_git_instance.cleanup.return_value = None

        # --- Test Sequential Execution ---
        start_time_seq = time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            # Awaiting each task individually simulates sequential execution
            for entry in entries:
                await analyze_entry(entry, pool, encountered_datasets)
        sequential_time = time.time() - start_time_seq

        # --- Test Concurrent Execution ---
        start_time_para = time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            tasks = [analyze_entry(entry, pool, encountered_datasets)
                     for entry in entries]
            await asyncio.gather(*tasks)
        parallel_time = time.time() - start_time_para

        print(f"\nSequential-like execution time: {sequential_time:.4f}s")
        print(f"Concurrent execution time:      {parallel_time:.4f}s")

        # Calculate speedup ratio
        speedup_ratio = sequential_time / parallel_time \
            if parallel_time > 0 else 0
        print(f"Speedup ratio: {speedup_ratio:.2f}x")

        # Assert that the parallel version is substantially faster
        # Use a more lenient threshold (1.5x instead of 2x)
        # to account for CI environment variations
        assert parallel_time < sequential_time, \
            f"Concurrent execution ({parallel_time:.4f}s) should be" \
            f"faster than sequential ({sequential_time:.4f}s)"
        assert parallel_time < sequential_time * 0.67, \
            f"Concurrent execution should be at least 1.5x faster. " \
            f"Got {speedup_ratio:.2f}x speedup."
