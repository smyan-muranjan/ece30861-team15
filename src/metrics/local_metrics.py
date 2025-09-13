import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from src.api.GitClient import GitClient, CommitStats, CodeQualityStats, RampUpStats


class LocalMetricsCalculator:
    """
    Calculator for metrics that require local repository analysis.
    """

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the metrics calculator.
        
        :param github_token: GitHub personal access token
        """
        self.git_client = GitClient()

    def calculate_bus_factor(self, repo_path: str) -> tuple[float, int]:
        """
        Calculate bus factor metric.
        
        :param repo_path: Path to local repository
        :return: Tuple of (score, latency_ms)
        """
        start_time = time.time()
        
        try:
            commit_stats = self.git_client.analyze_commits(repo_path)
            latency_ms = int((time.time() - start_time) * 1000)
            
            logging.info(f"Bus factor calculated: {commit_stats.bus_factor}")
            return commit_stats.bus_factor, latency_ms
            
        except Exception as e:
            logging.error(f"Failed to calculate bus factor: {str(e)}")
            latency_ms = int((time.time() - start_time) * 1000)
            return 0.0, latency_ms

    def calculate_code_quality(self, repo_path: str) -> tuple[float, int]:
        """
        Calculate code quality metric.
        
        :param repo_path: Path to local repository
        :return: Tuple of (score, latency_ms)
        """
        start_time = time.time()
        
        try:
            quality_stats = self.git_client.analyze_code_quality(repo_path)
            latency_ms = int((time.time() - start_time) * 1000)
            
            logging.info(f"Code quality calculated: {quality_stats.code_quality_score}")
            return quality_stats.code_quality_score, latency_ms
            
        except Exception as e:
            logging.error(f"Failed to calculate code quality: {str(e)}")
            latency_ms = int((time.time() - start_time) * 1000)
            return 0.0, latency_ms

    def calculate_ramp_up_time(self, repo_path: str) -> tuple[float, int]:
        """
        Calculate ramp-up time metric.
        
        :param repo_path: Path to local repository
        :return: Tuple of (score, latency_ms)
        """
        start_time = time.time()
        
        try:
            ramp_up_stats = self.git_client.analyze_ramp_up_time(repo_path)
            latency_ms = int((time.time() - start_time) * 1000)
            
            logging.info(f"Ramp-up time calculated: {ramp_up_stats.ramp_up_score}")
            return ramp_up_stats.ramp_up_score, latency_ms
            
        except Exception as e:
            logging.error(f"Failed to calculate ramp-up time: {str(e)}")
            latency_ms = int((time.time() - start_time) * 1000)
            return 0.0, latency_ms

    def calculate_size_score(self, repo_path: str) -> tuple[Dict[str, float], int]:
        """
        Calculate size score for different hardware platforms.
        
        :param repo_path: Path to local repository
        :return: Tuple of (size_scores_dict, latency_ms)
        """
        start_time = time.time()
        
        try:
            size_scores = self.git_client.get_repository_size(repo_path)
            latency_ms = int((time.time() - start_time) * 1000)
            
            logging.info(f"Size scores calculated: {size_scores}")
            return size_scores, latency_ms
            
        except Exception as e:
            logging.error(f"Failed to calculate size score: {str(e)}")
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                'raspberry_pi': 0.0,
                'jetson_nano': 0.0,
                'desktop_pc': 0.0,
                'aws_server': 0.0
            }, latency_ms

    def analyze_repository(self, url: str) -> Dict[str, Any]:
        """
        Analyze a repository and return all local metrics.
        
        :param url: Repository URL
        :return: Dictionary containing all calculated metrics
        """
        logging.info(f"Starting local analysis for: {url}")
        
        # Clone the repository
        repo_path = self.git_client.clone_repository(url)
        if not repo_path:
            logging.error(f"Failed to clone repository: {url}")
            return self._get_default_metrics()
        
        try:
            # Calculate all metrics
            bus_factor, bus_factor_latency = self.calculate_bus_factor(repo_path)
            code_quality, code_quality_latency = self.calculate_code_quality(repo_path)
            ramp_up_time, ramp_up_time_latency = self.calculate_ramp_up_time(repo_path)
            size_scores, size_score_latency = self.calculate_size_score(repo_path)
            
            return {
                'bus_factor': bus_factor,
                'bus_factor_latency': bus_factor_latency,
                'code_quality': code_quality,
                'code_quality_latency': code_quality_latency,
                'ramp_up_time': ramp_up_time,
                'ramp_up_time_latency': ramp_up_time_latency,
                'size_score': size_scores,
                'size_score_latency': size_score_latency,
            }
            
        finally:
            # Clean up the cloned repository
            self.git_client.cleanup()

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when analysis fails."""
        return {
            'bus_factor': 0.0,
            'bus_factor_latency': 0,
            'code_quality': 0.0,
            'code_quality_latency': 0,
            'ramp_up_time': 0.0,
            'ramp_up_time_latency': 0,
            'size_score': {
                'raspberry_pi': 0.0,
                'jetson_nano': 0.0,
                'desktop_pc': 0.0,
                'aws_server': 0.0
            },
            'size_score_latency': 0,
        }
