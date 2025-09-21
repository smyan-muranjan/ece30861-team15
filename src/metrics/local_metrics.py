import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, Optional

from src.api.git_client import GitClient
from src.metric_inputs.bus_factor_input import BusFactorInput
from src.metric_inputs.code_quality_input import CodeQualityInput
from src.metric_inputs.size_input import SizeInput
from src.metrics.bus_factor_metric import BusFactorMetric
from src.metrics.code_quality_metric import CodeQualityMetric
from src.metrics.size_metric import SizeMetric


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

        # Initialize metric instances
        self.bus_factor_metric = BusFactorMetric(self.git_client)
        self.code_quality_metric = CodeQualityMetric(self.git_client)
        self.size_metric = SizeMetric(self.git_client)

    async def _run_cpu_bound(self, func, *args) -> Any:
        """
        Runs a CPU-bound function in the process pool and measures latency.
        """
        loop = asyncio.get_running_loop()
        start_time = time.time()

        # For async functions, run them directly in the current event loop
        if asyncio.iscoroutinefunction(func):
            result = await func(*args)
        else:
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
            bus_factor_task = self._run_cpu_bound(
                self.bus_factor_metric.calculate,
                BusFactorInput(repo_url=repo_path))
            code_quality_task = self._run_cpu_bound(
                self.code_quality_metric.calculate,
                CodeQualityInput(repo_url=repo_path))
            # ramp_up_task = self._run_cpu_bound(
            #     self.ramp_up_time_metric.calculate, repo_path)
            size_task = self._run_cpu_bound(
                self.size_metric.calculate,
                SizeInput(repo_url=repo_path))

            (bus_factor_score, bus_lat), \
                (code_quality_score, qual_lat), \
                (size_score, size_lat) = \
                await asyncio.gather(bus_factor_task,
                                     code_quality_task,
                                     size_task)

            return {
                'bus_factor': bus_factor_score,
                'bus_factor_latency': bus_lat,
                'code_quality': code_quality_score,
                'code_quality_latency': qual_lat,
                # 'ramp_up_time': ramp_up_score,
                # 'ramp_up_time_latency': ramp_lat,
                'size_score': size_score,
                'size_score_latency': size_lat,
            }
        finally:
            self.git_client.cleanup()

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Returns a default metric structure on failure."""
        return {
            'bus_factor': 0.0, 'bus_factor_latency': 0,
            'code_quality': 0.0, 'code_quality_latency': 0,
            # 'ramp_up_time': 0.0, 'ramp_up_time_latency': 0,
            'size_score': {}, 'size_score_latency': 0,
        }
