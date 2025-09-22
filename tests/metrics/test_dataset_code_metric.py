import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.metric_inputs.dataset_code_input import DatasetCodeInput
from src.metrics.dataset_code_metric import DatasetCodeMetric


class TestDatasetCodeMetric:
    def setup_method(self):
        self.metric = DatasetCodeMetric()

    @pytest.mark.asyncio
    async def test_calculate_both_dataset_and_training_code(self):
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
        # Model Training

        ## Dataset
        This model was trained on the IMDB dataset available at:
        https://huggingface.co/datasets/imdb

        ## Training
        Run the training script with: python train.py
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "train.py").touch()
            (Path(temp_dir) / "model.py").touch()
            (Path(temp_dir) / "utils.py").touch()

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 1.0

    @pytest.mark.asyncio
    async def test_calculate_only_dataset_info(self):
        mock_git_client = Mock()

        mock_git_client.read_readme.return_value = """
        # Model Documentation

        ## Data Source
        The training data comes from the Kaggle competition:
        https://www.kaggle.com/competitions/sentiment-analysis

        ## Usage
        This model can be used for sentiment analysis.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files (no training scripts)
            (Path(temp_dir) / "model.py").touch()
            (Path(temp_dir) / "inference.py").touch()
            (Path(temp_dir) / "utils.py").touch()

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.5

    @pytest.mark.asyncio
    async def test_calculate_only_training_code(self):
        """Test case where only training code is present."""
        mock_git_client = Mock()

        mock_git_client.read_readme.return_value = """
        # Model Implementation

        This is a machine learning model for classification.

        ## Installation
        pip install -r requirements.txt
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files with training script
            (Path(temp_dir) / "finetune.py").touch()
            (Path(temp_dir) / "model.py").touch()
            (Path(temp_dir) / "data_loader.py").touch()

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.5

    @pytest.mark.asyncio
    async def test_calculate_neither_dataset_nor_training_code(self):
        mock_git_client = Mock()

        mock_git_client.read_readme.return_value = """
        # Model Documentation

        This is a pre-trained model for text classification.

        ## Usage
        ```python
        from transformers import pipeline
        classifier = pipeline("text-classification")
        ```
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files (no training scripts)
            (Path(temp_dir) / "model.py").touch()
            (Path(temp_dir) / "inference.py").touch()
            (Path(temp_dir) / "requirements.txt").touch()

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_with_config_file_dataset_info(self):
        """Test case where dataset info is found in config.json file."""
        mock_git_client = Mock()

        mock_git_client.read_readme.return_value = """
        # Model Documentation

        This is a machine learning model.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files (no training scripts)
            (Path(temp_dir) / "model.py").touch()
            (Path(temp_dir) / "inference.py").touch()

            config_content = """
            {
                "model_name": "bert-base-uncased",
                "dataset": "https://huggingface.co/datasets/glue",
                "training_data": "https://www.kaggle.com/datasets/sentiment140"
            }
            """

            metric = DatasetCodeMetric(mock_git_client)

            with patch.object(metric,
                              '_read_config_file',
                              return_value=config_content
                              ):
                result = await metric.calculate(
                    DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.5

    @pytest.mark.asyncio
    async def test_calculate_various_training_script_patterns(self):
        """Test detection of various training script naming patterns."""
        mock_git_client = Mock()

        mock_git_client.read_readme.return_value = """
        # Model Documentation

        This is a machine learning model.
        """

        test_cases = [
            ('train.py', 1.0),
            ('finetune.py', 1.0),
            ('training.py', 1.0),
            ('train_model.py', 1.0),
            ('fine_tune.py', 1.0),
            ('run_training.py', 1.0),
            ('model.py', 0.0),
            ('inference.py', 0.0)
        ]

        for filename, expected_score in test_cases:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test file
                (Path(temp_dir) / filename).touch()

                metric = DatasetCodeMetric(mock_git_client)
                result = await metric.calculate(
                    DatasetCodeInput(repo_url=temp_dir))

                expected = 0.5 if expected_score == 1.0 else 0.0
                assert result == expected, f"Failed for filename: {filename}"

    @pytest.mark.asyncio
    async def test_calculate_various_dataset_patterns(self):
        """Test detection of various dataset reference patterns."""
        mock_git_client = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file (no training scripts)
            (Path(temp_dir) / "model.py").touch()

            test_cases = [
                ("Dataset: https://huggingface.co/datasets/imdb", 0.5),
                ("Training data: kaggle.com/competitions/sentiment", 0.5),
                ("Data source: zenodo.org/record/12345", 0.5),
                ("Download from: figshare.com/articles/dataset", 0.5),
                ("No dataset information here", 0.0),
                ("This is just regular text", 0.0)
            ]

            for readme_content, expected_score in test_cases:
                mock_git_client.read_readme.return_value = readme_content

                metric = DatasetCodeMetric(mock_git_client)
                result = await metric.calculate(
                    DatasetCodeInput(repo_url=temp_dir))

                assert result == expected_score, \
                    f"Failed for content: {readme_content}"

    @pytest.mark.asyncio
    async def test_calculate_empty_readme_and_no_files(self):
        """Test case with empty README and no Python files."""
        mock_git_client = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory (no files)
            mock_git_client.read_readme.return_value = ""

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_readme_none(self):
        """Test case where README read returns None."""
        mock_git_client = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock None README
            mock_git_client.read_readme.return_value = None

            # Create test file (no training scripts)
            (Path(temp_dir) / "model.py").touch()

            metric = DatasetCodeMetric(mock_git_client)
            result = await metric.calculate(
                DatasetCodeInput(repo_url=temp_dir))

            assert result == 0.0
