import os
import re
from pathlib import Path
from typing import Any, Optional

from src.api.git_client import GitClient
from src.metric_inputs.dataset_code_input import DatasetCodeInput
from src.metrics.metric import Metric


class DatasetCodeMetric(Metric):
    def __init__(self, git_client: Optional[GitClient] = None):
        self.git_client = git_client or GitClient()
        self.training_script_patterns = [
            r'train\.py$',
            r'finetune\.py$',
            r'training\.py$',
            r'train_.*\.py$',
            r'fine_tune\.py$',
            r'fine-tune\.py$',
            r'model_train\.py$',
            r'train_model\.py$',
            r'run_training\.py$',
            r'train_script\.py$'
        ]

        self.dataset_patterns = [
            r'dataset[s]?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'training\s+data[s]?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'data[s]?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'https?://[^\s]+dataset[^\s]*',
            r'https?://[^\s]+data[^\s]*',
            r'huggingface\.co/[^\s]+',
            r'kaggle\.com/[^\s]+',
            r'github\.com/[^\s]+data[^\s]*',
            r'zenodo\.org/[^\s]+',
            r'figshare\.com/[^\s]+'
        ]

    async def calculate(self, metric_input: Any) -> float:
        assert isinstance(metric_input, DatasetCodeInput)
        has_dataset_info = self._check_dataset_info(metric_input.repo_url)
        has_training_code = self._check_training_code(metric_input.repo_url)
        score = (has_dataset_info + has_training_code) / 2.0
        return score

    def _check_dataset_info(self, repo_url: str) -> int:
        readme_content = self.git_client.read_readme(repo_url)
        if readme_content and self._find_dataset_references(readme_content):
            return 1

        config_content = self._read_config_file(repo_url)
        if config_content and self._find_dataset_references(config_content):
            return 1

        return 0

    def _check_training_code(self, repo_url: str) -> int:
        try:
            # Check if path exists first
            if not os.path.exists(repo_url):
                return 0

            repo_path_obj = Path(repo_url)
            python_files = list(repo_path_obj.rglob("*.py"))

            if not python_files:
                return 0

            for file_path in python_files:
                filename = os.path.basename(file_path).lower()
                for pattern in self.training_script_patterns:
                    if re.search(pattern, filename):
                        return 1

            return 0
        except Exception:
            return 0

    def _find_dataset_references(self, content: str) -> bool:
        if not content:
            return False

        content_lower = content.lower()

        for pattern in self.dataset_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True

        dataset_keywords = [
            'training data', 'data source', 'data link',
            'download data', 'data url', 'data repository', 'data file',
            'huggingface', 'kaggle', 'zenodo', 'figshare'
        ]

        for keyword in dataset_keywords:
            if keyword in content_lower:
                return True

        dataset_context_patterns = [
            r'dataset[s]?\s*[:=]',
            r'dataset[s]?\s+is\s+',
            r'dataset[s]?\s+available',
            r'dataset[s]?\s+from',
            r'dataset[s]?\s+at',
            r'dataset[s]?\s+can\s+be',
            r'using\s+dataset[s]?',
            r'train[ed]?\s+on\s+dataset[s]?',
            r'dataset[s]?\s+used',
            r'dataset[s]?\s+for\s+training'
        ]

        for pattern in dataset_context_patterns:
            if re.search(pattern, content_lower):
                return True

        return False

    def _read_config_file(self, repo_url: str) -> Optional[str]:
        try:
            config_path = os.path.join(repo_url, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return f.read()

            config_files = ['config.yaml',
                            'config.yml',
                            'configuration.json',
                            'settings.json']
            for config_file in config_files:
                config_path = os.path.join(repo_url, config_file)
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return f.read()

            return None
        except Exception:
            return None
