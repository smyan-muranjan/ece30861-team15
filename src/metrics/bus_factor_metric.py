from typing import Any, Optional

from src.api.git_client import GitClient
from src.metrics.metric import Metric


class BusFactorMetric(Metric):
    def __init__(self, git_client: Optional[GitClient] = None):
        self.git_client = git_client or GitClient()
    
    async def calculate(self, metric_input: Any) -> float:
        assert isinstance(metric_input, str)

        commit_stats = self.git_client.analyze_commits(metric_input)
        if not commit_stats or commit_stats.total_commits == 0:
            return 0.0

        concentration = sum(
            (count / commit_stats.total_commits) ** 2
            for count in commit_stats.contributors.values()
        )
        return max(0.0, 1.0 - concentration)
