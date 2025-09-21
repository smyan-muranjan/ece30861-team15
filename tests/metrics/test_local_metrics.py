from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.metrics.local_metrics import LocalMetricsCalculator


@pytest.fixture
def mock_git_client():
    """
    Fixture to create a properly configured mock GitClient.
    It patches the GitClient inside the local_metrics module,
    ensuring our calculator uses the mock.
    """
    with patch('src.metrics.local_metrics.GitClient') as MockGitClient:
        mock_instance = MockGitClient.return_value
        yield mock_instance


@pytest.mark.asyncio
async def test_analyze_repository_success(mock_git_client):
    """
    Tests the successful analysis path, ensuring positive scores are returned
    when mock data indicates a high-quality repository.
    """
    # Arrange: Configure mock data for a high-quality repository
    mock_git_client.clone_repository.return_value = "/tmp/fake/repo"
    mock_git_client.analyze_commits.return_value = MagicMock(
        total_commits=100, contributors={'author1': 50, 'author2': 50}
    )
    mock_git_client.analyze_code_quality.return_value = MagicMock(
        lint_errors=0, has_tests=True
    )
    mock_git_client.analyze_ramp_up_time.return_value = MagicMock(
        readme_quality=1.0, has_examples=True, has_dependencies=True
    )

    # **FIXED**: Use ThreadPoolExecutor in this test to
    # avoid pickling MagicMock objects.
    # Application's use of ProcessPoolExecutor is still correct for production.
    with ThreadPoolExecutor() as pool:
        calculator = LocalMetricsCalculator(pool)
        # Act
        result = await calculator.analyze_repository("http://test.url")

    # Assert
    assert result['bus_factor'] == 0.5
    assert result['code_quality'] == 1.0
    mock_git_client.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_repository_clone_fails(mock_git_client):
    """
    Tests the failure path where the git clone operation returns None.
    """
    # Arrange
    mock_git_client.clone_repository.return_value = None

    # Using ThreadPoolExecutor here as well for consistency
    with ThreadPoolExecutor() as pool:
        calculator = LocalMetricsCalculator(pool)
        # Act
        result = await calculator.analyze_repository("http://invalid.url")

    # Assert
    assert result['bus_factor'] == 0.0
    assert result['code_quality'] == 0.0
    assert result['ramp_up_time'] == 0.0
    mock_git_client.cleanup.assert_not_called()


def test_calculate_code_quality_perfect_score():
    """
    Tests the code quality calculation for a project with no lint errors
    and a full test suite, which must equal 1.0.
    """
    # Arrange
    with patch('src.metrics.local_metrics.GitClient') as MockGitClient:
        mock_instance = MockGitClient.return_value
        mock_instance.analyze_code_quality.return_value = MagicMock(
            lint_errors=0, has_tests=True
        )
        calculator = LocalMetricsCalculator(MagicMock())
        # Act
        score = calculator.calculate_code_quality("/tmp/fake/repo")

    # Assert
    assert score == 1.0
