import math
from typing import Dict

from huggingface_hub import HfApi, hf_hub_download

from src.constants import MAX_DATASET_DOWNLOADS, MAX_DATASET_LIKES
from src.models.dataset_stats import DatasetStats


class HuggingFaceClient:
    """
    Minimal wrapper for interacting with the Hugging Face Hub.
    No authentication (token) required — works only for public repos.
    """

    def __init__(self):
        self.api = HfApi()

    def normalize_log(self, value: int, max_value: int) -> float:
        """
        Normalize a value to a [0, 1] scale using logarithmic normalization.
        """
        if value <= 0:
            return 0.0
        return min(math.log1p(value) / math.log1p(max_value), 1.0)

    def get_model_info(self, repo_id: str) -> Dict:
        """
        Get metadata for a model repo (downloads, likes, tags, etc.).

        :param repo_id: "namespace/repo_name"
        :return: dict containing model info
        """
        return self.api.model_info(repo_id)

    def get_dataset_info(self, repo_id: str) -> DatasetStats:
        """
        Get number of likes and downloads for a dataset repo.

        :param repo_id: "namespace/dataset_name"
        :return: DatasetStats with likes and downloads
        """
        info = self.api.dataset_info(repo_id)
        normalized_likes = self.normalize_log(info.likes, MAX_DATASET_LIKES)
        normalized_downloads = self.normalize_log(info.downloads, MAX_DATASET_DOWNLOADS)
        return DatasetStats(normalized_likes=normalized_likes, normalized_downloads=normalized_downloads)

    def download_file(
        self, repo_id: str, filename: str, local_dir: str = "./"
    ) -> str:
        """
        Download a file from a repo (e.g. README.md).

        :param repo_id: "namespace/repo_name"
        :param filename: path to file in the repo
        :param local_dir: directory to download to
        :return: local path to downloaded file
        """
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
            )
