from unittest.mock import Mock

import pytest

from src.metric_inputs.license_input import LicenseInput
from src.metrics.license_metric import LicenseMetric


class TestLicenseMetric:
    def setup_method(self):
        self.metric = LicenseMetric()

    @pytest.mark.asyncio
    async def test_calculate_permissive_license_mit(self):
        """Test MIT license gets 1.0 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
# Project Title

## License

This project is licensed under the MIT License.
        """.strip()

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_calculate_lgpl_license(self):
        """Test LGPL license gets 0.5 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
# Project Title

## License

This project is licensed under the GNU Lesser General Public License v2.1.
        """.strip()

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 0.5

    @pytest.mark.asyncio
    async def test_calculate_copyleft_license_gpl(self):
        """Test GPL license gets 0.1 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
# Project Title

## License

This project is licensed under the GNU General Public License v3.
        """.strip()

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 0.1

    @pytest.mark.asyncio
    async def test_calculate_no_license_section(self):
        """Test no license section gets 0.0 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
# Project Title

## Installation

Run `pip install package`.
        """.strip()

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_no_readme(self):
        """Test no README gets 0.0 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = None

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_unknown_license(self):
        """Test unknown license gets 0.0 score."""
        mock_git_client = Mock()
        mock_git_client.read_readme.return_value = """
# Project Title

## License

This project uses a custom proprietary license.
        """.strip()

        metric = LicenseMetric(mock_git_client)
        result = await metric.calculate(LicenseInput(repo_url="/test/repo"))

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_invalid_type(self):
        """Test that invalid input type raises AssertionError."""
        with pytest.raises(AssertionError):
            await self.metric.calculate("invalid_input")

    def test_score_license_all_categories(self):
        """Test license scoring for all categories."""
        # Permissive licenses (1.0)
        assert self.metric._score_license("MIT License") == 1.0
        assert self.metric._score_license("Apache 2.0") == 1.0

        # LGPL licenses (0.5)
        assert self.metric._score_license("LGPL v2.1") == 0.5

        # Copyleft licenses (0.1)
        assert self.metric._score_license("GPL v3") == 0.1

        # Unknown licenses (0.0)
        assert self.metric._score_license("Custom License") == 0.0
