import os
import tempfile
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime, timedelta

from git import Repo


@dataclass
class CommitStats:
    """Statistics about commits in a repository."""
    total_commits: int
    contributors: Dict[str, int]  # author -> commit count
    bus_factor: float


class GitClient:
    """
    Client for cloning and analyzing Git repositories.
    """

    def __init__(self):
        """Initialize Git client."""
        self.temp_dirs: List[str] = []  # Track temp dirs for cleanup

    def clone_repository(self, url: str) -> Optional[str]:
        """
        Clone a repository to a temporary directory.
        
        :param url: Git repository URL
        :return: Path to cloned repository, or None if cloning failed
        """
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="model_analysis_")
            self.temp_dirs.append(temp_dir)
            
            logging.info(f"Cloning repository: {url}")
            
            # Clone the repository
            repo = Repo.clone_from(url, temp_dir)
            logging.info(f"Successfully cloned to: {temp_dir}")
            
            return temp_dir
            
        except Exception as e:
            logging.error(f"Failed to clone repository {url}: {str(e)}")
            return None

    def analyze_commits(self, repo_path: str) -> CommitStats:
        """
        Analyze commit history to calculate bus factor.
        
        :param repo_path: Path to local repository
        :return: CommitStats object
        """
        try:
            repo = Repo(repo_path)
            
            # Get commits from the last 365 days
            since_date = datetime.now() - timedelta(days=365)
            commits = list(repo.iter_commits(since=since_date))
            
            # Count commits by author
            contributors = {}
            for commit in commits:
                author = commit.author.name
                contributors[author] = contributors.get(author, 0) + 1
            
            # Calculate bus factor using Herfindahl-Hirschman Index
            total_commits = len(commits)
            if total_commits == 0:
                return CommitStats(total_commits=0, contributors={}, bus_factor=0.0)
            
            # Get top 10 contributors
            top_contributors = sorted(contributors.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate concentration index
            concentration = sum((count / total_commits) ** 2 for _, count in top_contributors)
            bus_factor = 1.0 - concentration  # Higher is better (more distributed)
            
            return CommitStats(
                total_commits=total_commits,
                contributors=dict(top_contributors),
                bus_factor=max(0.0, min(1.0, bus_factor))
            )
            
        except Exception as e:
            logging.error(f"Failed to analyze commits: {str(e)}")
            return CommitStats(total_commits=0, contributors={}, bus_factor=0.0)

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to clean up {temp_dir}: {str(e)}")
        self.temp_dirs.clear()
