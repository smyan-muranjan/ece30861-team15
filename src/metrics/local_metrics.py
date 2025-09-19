import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, Optional

from src.api.git_client import GitClient


class LocalMetricsCalculator:
    """
    Calculator for metrics that require local repository analysis,
    optimized for concurrent execution and aligned with the project plan.
    """

    def __init__(
        self,
        process_pool: ProcessPoolExecutor,
        github_token: Optional[str] = None
    ):
        self.git_client = GitClient()
        self.process_pool = process_pool
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

    async def _run_cpu_bound(self, func, *args) -> Any:
        """
        Runs a CPU-bound function in the process pool and measures latency.
        """
        loop = asyncio.get_running_loop()
        start_time = time.time()
        result = await loop.run_in_executor(self.process_pool, func, *args)
        latency = int((time.time() - start_time) * 1000)
        return result, latency

    async def analyze_repository(self, url: str) -> Dict[str, Any]:
        """
        Clones and analyzes a repository, running all metric calculations
        in parallel according to the project plan's operationalization.
        """
        logging.info(f"Starting async analysis for: {url}")
        loop = asyncio.get_running_loop()

        repo_path = await loop.run_in_executor(
            self.thread_pool, self.git_client.clone_repository, url
        )

        if not repo_path:
            logging.error(f"Failed to clone repository: {url}")
            return self._get_default_metrics()

        try:
            bus_factor_task = self._run_cpu_bound(self.calculate_bus_factor,
                                                  repo_path)
            code_quality_task = self._run_cpu_bound(
                self.calculate_code_quality,
                repo_path)
            ramp_up_task = self._run_cpu_bound(self.calculate_ramp_up_time,
                                               repo_path)

            (bus_factor_score, bus_lat), \
                (code_quality_score, qual_lat), \
                (ramp_up_score, ramp_lat) = \
                await asyncio.gather(bus_factor_task,
                                     code_quality_task,
                                     ramp_up_task)

            return {
                'bus_factor': bus_factor_score,
                'bus_factor_latency': bus_lat,
                'code_quality': code_quality_score,
                'code_quality_latency': qual_lat,
                'ramp_up_time': ramp_up_score,
                'ramp_up_time_latency': ramp_lat,
            }
        finally:
            self.git_client.cleanup()

    # --- Metric Calculation Methods based on Project Plan ---

    def calculate_bus_factor(self, repo_path: str) -> float:
        """Calculates bus factor based on commit concentration."""
        commit_stats = self.git_client.analyze_commits(repo_path)
        if not commit_stats or commit_stats.total_commits == 0:
            return 0.0

        # **FIXED**: The formula in the plan is Score = 1 - Sum(pi^2),
        # which is correct for bus factor.
        # A lower concentration (more distributed authors)
        # results in a higher score.
        concentration = sum(
            (count / commit_stats.total_commits) ** 2
            for count in commit_stats.contributors.values()
        )
        return max(0.0, 1.0 - concentration)

    def calculate_code_quality(self, repo_path: str) -> float:
        """Calculates code quality from linter errors and test presence."""
        quality_stats = self.git_client.analyze_code_quality(repo_path)
        # Score component for linter errors, as per the plan [cite: 65]
        lint_score = max(0.0, 1.0 - (quality_stats.lint_errors * 0.05))
        has_tests_score = 1.0 if quality_stats.has_tests else 0.0

        # **FIXED**: Using the correct weighted formula
        # from project plan [cite: 65]
        return (0.6 * lint_score) + (0.4 * has_tests_score)

    def calculate_ramp_up_time(self, repo_path: str) -> float:
        """Calculates ramp-up time from docs, examples, and dependencies."""
        ramp_up_stats = self.git_client.analyze_ramp_up_time(repo_path)
        # Placeholder for the required LLM analysis of the README [cite: 60]
        readme_llm_score = ramp_up_stats.readme_quality
        has_examples_score = 1.0 if ramp_up_stats.has_examples else 0.0
        has_deps_score = 1.0 if ramp_up_stats.has_dependencies else 0.0

        # **FIXED**: Using the correct weighted formula
        # from project plan [cite: 60]
        score = (0.6 * readme_llm_score) + \
                (0.25 * has_examples_score) + \
                (0.15 * has_deps_score)
        return score

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Returns a default metric structure on failure."""
        return {
            'bus_factor': 0.0, 'bus_factor_latency': 0,
            'code_quality': 0.0, 'code_quality_latency': 0,
            'ramp_up_time': 0.0, 'ramp_up_time_latency': 0,
        }
