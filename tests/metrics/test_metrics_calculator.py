from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.metrics.metrics_calculator import (MetricsCalculator,
                                            extract_hf_repo_id,
                                            is_code_repository, is_dataset_url,
                                            is_model_url)


@pytest.fixture
def mock_clients():
    """
    Fixture to create properly configured mock clients.
    It patches all API clients inside the metrics_calculator module,
    ensuring our calculator uses the mocks.
    """
    with patch('src.metrics.metrics_calculator.GitClient') as MockGitClient, \
         patch(
             'src.metrics.metrics_calculator.GenAIClient'
             ) as MockGenAIClient, \
         patch(
             'src.metrics.metrics_calculator.HuggingFaceClient'
             ) as MockHuggingFaceClient:

        mock_git = MockGitClient.return_value
        mock_genai = MockGenAIClient.return_value
        mock_hf = MockHuggingFaceClient.return_value

        yield {
            'git': mock_git,
            'genai': mock_genai,
            'hf': mock_hf
        }


@pytest.mark.asyncio
async def test_analyze_repository_success(mock_clients):
    """
    Tests the successful analysis path, ensuring positive scores are returned
    when mock data indicates a high-quality repository.
    """
    # Arrange: Configure mock data for a high-quality repository
    mock_git = mock_clients['git']
    mock_genai = mock_clients['genai']
    mock_hf = mock_clients['hf']

    mock_git.clone_repository.return_value = "/tmp/fake/repo"
    mock_git.analyze_commits.return_value = MagicMock(
        total_commits=100, contributors={'author1': 50, 'author2': 50}
    )
    mock_git.analyze_code_quality.return_value = MagicMock(
        lint_errors=0, has_tests=True
    )
    mock_git.read_readme.return_value = """
# Project Title

## License

This project is licensed under the MIT License.
    """.strip()
    mock_git.get_repository_size.return_value = {
        'raspberry_pi': 1.0,
        'jetson_nano': 1.0,
        'desktop_pc': 1.0,
        'aws_server': 1.0
    }

    # Mock GenAI client responses (async methods)
    mock_genai.get_performance_claims = AsyncMock(return_value={
        "has_metrics": 1, "mentions_benchmarks": 1
    })
    mock_genai.get_readme_clarity = AsyncMock(return_value=0.8)

    # Mock HuggingFace client responses (async methods)
    mock_hf.get_dataset_info = AsyncMock(return_value={
        'likes': 100, 'downloads': 1000
    })

    # **FIXED**: Use ThreadPoolExecutor in this test to
    # avoid pickling MagicMock objects.
    # Application's use of
    # ProcessPoolExecutor is still correct for production.
    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)
        # Act
        result = await calculator.analyze_repository("http://test.url")

    # Assert
    assert result['bus_factor'] == 0.5
    assert result['code_quality'] == 1.0
    assert result['license'] == 1.0
    assert 'performance_claims' in result
    assert 'dataset_quality' in result
    mock_git.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_repository_clone_fails(mock_clients):
    """
    Tests the failure path where the git clone operation returns None.
    """
    # Arrange
    mock_git = mock_clients['git']
    mock_git.clone_repository.return_value = None

    # Using ThreadPoolExecutor here as well for consistency
    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)
        # Act
        result = await calculator.analyze_repository("http://invalid.url")

    # Assert
    assert result['bus_factor'] == 0.0
    assert result['code_quality'] == 0.0
    assert result['license'] == 0.0
    # assert result['ramp_up_time'] == 0.0
    mock_git.cleanup.assert_not_called()


def test_extract_hf_repo_id():
    """Tests the extract_hf_repo_id function with various URLs."""
    # Test dataset URLs
    assert extract_hf_repo_id("https://huggingface.co/datasets/allenai/c4") \
        == "allenai/c4"

    # Test model URLs
    assert extract_hf_repo_id(
        "https://huggingface.co/ibm-granite/granite-docling-258M"
                             ) == "ibm-granite/granite-docling-258M"
    assert extract_hf_repo_id("https://huggingface.co/user/model") \
        == "user/model"

    # Test invalid URLs
    with pytest.raises(ValueError):
        extract_hf_repo_id("https://github.com/user/repo")
    with pytest.raises(ValueError):
        extract_hf_repo_id("https://huggingface.co/spaces/user/space")


def test_is_code_repository():
    """Tests the is_code_repository function with various URLs."""
    # Valid GitHub URLs
    assert is_code_repository("https://github.com/user/repo")
    assert is_code_repository("https://github.com/user/repo.git")

    # Valid GitLab URLs
    assert is_code_repository("https://gitlab.com/user/repo")
    assert is_code_repository("https://gitlab.com/user/repo.git")

    # Valid HuggingFace Spaces URLs
    assert is_code_repository("https://huggingface.co/spaces/user/space")

    # Invalid URLs
    assert not is_code_repository("https://huggingface.co/models/user/model")
    assert not is_code_repository("https://huggingface.co/\
                                  datasets/user/dataset")
    assert not is_code_repository("https://example.com")
    assert not is_code_repository("")
    assert not is_code_repository(None)


def test_is_dataset_url():
    """Tests the is_dataset_url function with various URLs."""
    # Valid HuggingFace dataset URLs
    assert is_dataset_url("https://huggingface.co/datasets/user/dataset")
    assert is_dataset_url("https://huggingface.co/datasets/glue")

    # Valid other dataset URLs
    assert is_dataset_url("https://image-net.org")
    assert is_dataset_url("https://kaggle.com/dataset")
    assert is_dataset_url("https://archive.ics.uci.edu/ml")

    # URLs with /datasets/ in path
    assert is_dataset_url("https://example.com/datasets/data")

    # Invalid URLs
    assert not is_dataset_url("https://huggingface.co/models/user/model")
    assert not is_dataset_url("https://github.com/user/repo")
    assert not is_dataset_url("")
    assert not is_dataset_url(None)


def test_is_model_url():
    """Tests the is_model_url function with various URLs."""
    # Valid HuggingFace model URLs
    assert is_model_url("https://huggingface.co/user/model")
    assert is_model_url("https://huggingface.co/google-bert/bert-base-uncased")

    # Invalid URLs
    assert not is_model_url("https://huggingface.co/datasets/user/dataset")
    assert not is_model_url("https://huggingface.co/spaces/user/space")
    assert not is_model_url("https://github.com/user/repo")
    assert not is_model_url("")
    assert not is_model_url(None)


@pytest.mark.asyncio
async def test_analyze_entry_with_code_link(mock_clients):
    """Tests analyze_entry with a code link."""
    mock_git = mock_clients['git']
    mock_genai = mock_clients['genai']
    mock_hf = mock_clients['hf']

    # Mock successful repository analysis
    mock_git.clone_repository.return_value = "/tmp/fake/repo"
    mock_git.analyze_commits.return_value = MagicMock(
        total_commits=100, contributors={'author1': 50, 'author2': 50}
    )
    mock_git.analyze_code_quality.return_value = MagicMock(
        lint_errors=0, has_tests=True
    )
    mock_git.read_readme.return_value = "# Project\n## License\nMIT"
    mock_git.get_repository_size.return_value = {
        'raspberry_pi': 1.0, 'jetson_nano': 1.0,
        'desktop_pc': 1.0, 'aws_server': 1.0
    }

    mock_genai.get_performance_claims = \
        AsyncMock(return_value={"has_metrics": 1})
    mock_genai.get_readme_clarity = AsyncMock(return_value=0.8)
    mock_hf.get_dataset_info = \
        MagicMock(return_value={'normalized_likes': 0.5,
                  'normalized_downloads': 0.7})

    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)
        result = await calculator.analyze_entry(
            "https://github.com/user/repo",
            "https://huggingface.co/datasets/user/dataset",
            "https://huggingface.co/user/model"
        )

    assert 'bus_factor' in result
    assert 'code_quality' in result
    assert 'license' in result
    assert 'dataset_quality' in result
    assert 'dataset_and_code_score' in result


@pytest.mark.asyncio
async def test_analyze_entry_no_repository(mock_clients):
    """Tests analyze_entry when no repository is available."""
    mock_git = mock_clients['git']
    mock_git.clone_repository.return_value = None

    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)
        result = await calculator.analyze_entry(
            None,  # No code link
            None,  # No dataset link
            "https://huggingface.co/user/model"  # Model URL (not a repo)
        )

    # Should return default metrics
    assert result['bus_factor'] == 0.0
    assert result['code_quality'] == 0.0
    assert result['license'] == 0.0
    assert result['dataset_and_code_score'] == 0.0


def test_calculate_dataset_and_code_score():
    """Tests _calculate_dataset_and_code_score method."""
    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)

        # Test with both code and dataset links
        score = calculator._calculate_dataset_and_code_score(
            "https://github.com/user/repo",
            "https://huggingface.co/datasets/user/dataset",
            {'code_quality': 0.8, 'dataset_quality': 0.9}
        )
        assert 0.0 <= score <= 1.0

        # Test with only code link
        score = calculator._calculate_dataset_and_code_score(
            "https://github.com/user/repo",
            None,
            {'code_quality': 0.8}
        )
        assert 0.0 <= score <= 1.0

        # Test with only dataset link
        score = calculator._calculate_dataset_and_code_score(
            None,
            "https://huggingface.co/datasets/user/dataset",
            {'dataset_quality': 0.9}
        )
        assert 0.0 <= score <= 1.0

        # Test with no links
        score = calculator._calculate_dataset_and_code_score(
            None,
            None,
            {}
        )
        assert score == 0.0


def test_get_default_metrics():
    """Tests _get_default_metrics method."""
    with ThreadPoolExecutor() as pool:
        calculator = MetricsCalculator(pool)
        metrics = calculator._get_default_metrics()

        # Should return all required metrics with default values
        expected_keys = [
            'bus_factor', 'bus_factor_latency',
            'code_quality', 'code_quality_latency',
            'license', 'license_latency',
            'ramp_up_time', 'ramp_up_time_latency',
            'performance_claims', 'performance_claims_latency',
            'size_score', 'size_score_latency',
            'dataset_quality', 'dataset_quality_latency'
        ]

        for key in expected_keys:
            assert key in metrics
            assert metrics[key] == 0.0 or metrics[key] == {}
