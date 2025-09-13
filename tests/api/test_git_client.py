import os
import tempfile
import shutil
import time
import stat
from pathlib import Path
import unittest
from git import Repo, Actor

from src.api.GitClient import GitClient, CommitStats


class TestGitClient(unittest.TestCase):
    """Test cases for GitClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.git_client = GitClient()
        self.temp_repo_path = None

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_repo_path and os.path.exists(self.temp_repo_path):
            self._force_remove_directory(self.temp_repo_path)
        self.git_client.cleanup()

    def _force_remove_directory(self, path):
        """Force remove directory with retries for Windows file locking issues."""
        def handle_remove_readonly(func, path, exc):
            """Handle readonly files on Windows."""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
                break
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Increasing delay
                else:
                    print(f"Warning: Could not remove {path} after {max_retries} attempts")

    def create_test_repo(self) -> str:
        """Create a test repository for testing."""
        self.temp_repo_path = tempfile.mkdtemp(prefix="test_repo_")
        repo = Repo.init(self.temp_repo_path)
        
        # Create a test file
        test_file = Path(self.temp_repo_path) / "test.py"
        test_file.write_text("print('Hello, World!')")
        
        # Make initial commit with a specific author
        default_author = Actor("DefaultAuthor", "default@test.com")
        repo.index.add(["test.py"])
        repo.index.commit("Initial commit", author=default_author)
        
        return self.temp_repo_path

    def test_analyze_commits(self):
        """Test commit analysis."""
        repo_path = self.create_test_repo()
        
        # Add more commits with different authors
        repo = Repo(repo_path)
        
        # Create commits with different authors
        test_file = Path(repo_path) / "test.py"
        test_file.write_text("print('Updated!')")
        repo.index.add(["test.py"])
        author1 = Actor("Author1", "author1@test.com")
        repo.index.commit("Update test file", author=author1)
        
        test_file.write_text("print('Updated again!')")
        repo.index.add(["test.py"])
        author2 = Actor("Author2", "author2@test.com")
        repo.index.commit("Another update", author=author2)
        
        commit_stats = self.git_client.analyze_commits(repo_path)
        
        self.assertIsInstance(commit_stats, CommitStats)
        self.assertGreaterEqual(commit_stats.total_commits, 3)  # Initial + 2 new commits
        self.assertGreaterEqual(len(commit_stats.contributors), 2)
        self.assertGreaterEqual(commit_stats.bus_factor, 0.0)
        self.assertLessEqual(commit_stats.bus_factor, 1.0)

    def test_analyze_commits_empty_repo(self):
        """Test commit analysis on empty repository."""
        empty_repo_path = tempfile.mkdtemp(prefix="empty_repo_")
        try:
            Repo.init(empty_repo_path)
            commit_stats = self.git_client.analyze_commits(empty_repo_path)
            
            self.assertEqual(commit_stats.total_commits, 0)
            self.assertEqual(commit_stats.bus_factor, 0.0)
        finally:
            shutil.rmtree(empty_repo_path)

    def test_analyze_commits_single_author(self):
        """Test commit analysis with single author (low bus factor)."""
        repo_path = self.create_test_repo()
        
        # Add more commits with same author as the initial commit
        repo = Repo(repo_path)
        
        test_file = Path(repo_path) / "test.py"
        same_author = Actor("DefaultAuthor", "default@test.com")  # Same as initial commit
        for i in range(5):
            test_file.write_text(f"print('Update {i}')")
            repo.index.add(["test.py"])
            repo.index.commit(f"Update {i}", author=same_author)
        
        commit_stats = self.git_client.analyze_commits(repo_path)
        
        self.assertIsInstance(commit_stats, CommitStats)
        self.assertGreaterEqual(commit_stats.total_commits, 6)  # Initial + 5 new commits
        self.assertEqual(len(commit_stats.contributors), 1)
        self.assertLess(commit_stats.bus_factor, 0.5)  # Should be low for single author

    def test_analyze_commits_multiple_authors(self):
        """Test commit analysis with multiple authors (high bus factor)."""
        repo_path = self.create_test_repo()
        
        # Add commits with different authors
        repo = Repo(repo_path)
        
        test_file = Path(repo_path) / "test.py"
        authors = [
            Actor("Author1", "author1@test.com"),
            Actor("Author2", "author2@test.com"),
            Actor("Author3", "author3@test.com")
        ]
        
        for i, author in enumerate(authors):
            test_file.write_text(f"print('Update by {author.name}')")
            repo.index.add(["test.py"])
            repo.index.commit(f"Update by {author.name}", author=author)
        
        commit_stats = self.git_client.analyze_commits(repo_path)
        
        self.assertIsInstance(commit_stats, CommitStats)
        self.assertGreaterEqual(commit_stats.total_commits, 4)  # Initial + 3 new commits
        self.assertGreaterEqual(len(commit_stats.contributors), 3)
        self.assertGreater(commit_stats.bus_factor, 0.5)  # Should be higher for multiple authors

    def test_cleanup(self):
        """Test cleanup functionality."""
        # Create some temp directories
        temp_dir1 = tempfile.mkdtemp(prefix="test_cleanup_1_")
        temp_dir2 = tempfile.mkdtemp(prefix="test_cleanup_2_")
        
        self.git_client.temp_dirs = [temp_dir1, temp_dir2]
        
        # Verify directories exist
        self.assertTrue(os.path.exists(temp_dir1))
        self.assertTrue(os.path.exists(temp_dir2))
        
        # Clean up
        self.git_client.cleanup()
        
        # Verify directories are removed
        self.assertFalse(os.path.exists(temp_dir1))
        self.assertFalse(os.path.exists(temp_dir2))
        self.assertEqual(len(self.git_client.temp_dirs), 0)

    def test_clone_repository_invalid_url(self):
        """Test cloning with invalid URL."""
        result = self.git_client.clone_repository("https://github.com/nonexistent/repo")
        self.assertIsNone(result)

    def test_analyze_commits_invalid_path(self):
        """Test commit analysis with invalid path."""
        commit_stats = self.git_client.analyze_commits("/nonexistent/path")
        
        self.assertIsInstance(commit_stats, CommitStats)
        self.assertEqual(commit_stats.total_commits, 0)
        self.assertEqual(commit_stats.bus_factor, 0.0)
        self.assertEqual(len(commit_stats.contributors), 0)


if __name__ == '__main__':
    unittest.main()
