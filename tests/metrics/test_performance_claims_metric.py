import pytest
from unittest.mock import AsyncMock, patch

from src.metrics.performance_claims_metric import PerformanceClaimsMetric
from src.metric_inputs.performance_input import PerformanceInput


class TestPerformanceClaimsMetric:
    def setup_method(self):
        self.metric = PerformanceClaimsMetric()

    @pytest.mark.asyncio
    async def test_calculate_with_valid_data(self):
        # Mock the GenAIClient
        mock_gen_ai_client = AsyncMock()
        mock_gen_ai_client.get_performance_claims.return_value = {
            "mentions_benchmarks": 0.8,
            "has_metrics": 0.6
        }

        # Patch the GenAIClient to use the mock
        with patch(
                "src.metrics.performance_claims_metric.GenAIClient",
                return_value=mock_gen_ai_client
                ):
            # Create test data
            metric_input = PerformanceInput(
                readme_text="This is a README with benchmarks and metrics."
                )

            # Call the calculate method
            result = await self.metric.calculate(metric_input)

            # Assert the result
            expected_result = (
                PerformanceClaimsMetric.HAS_BENCHMARKS_WEIGHT * 0.8 +
                PerformanceClaimsMetric.HAS_METRICS_WEIGHT * 0.6
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_calculate_with_empty_data(self):
        # Mock the GenAIClient
        mock_gen_ai_client = AsyncMock()
        mock_gen_ai_client.get_performance_claims.return_value = {
            "mentions_benchmarks": 0.0,
            "has_metrics": 0.0
        }

        # Patch the GenAIClient to use the mock
        with patch(
                "src.metrics.performance_claims_metric.GenAIClient",
                return_value=mock_gen_ai_client
                ):
            # Create test data
            metric_input = PerformanceInput(readme_text="")

            # Call the calculate method
            result = await self.metric.calculate(metric_input)

            # Assert the result
            expected_result = 0.0
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_calculate_with_partial_data(self):
        # Mock the GenAIClient
        mock_gen_ai_client = AsyncMock()
        mock_gen_ai_client.get_performance_claims.return_value = {
            "mentions_benchmarks": 0.5,
            "has_metrics": 0.0
        }

        # Patch the GenAIClient to use the mock
        with patch(
                "src.metrics.performance_claims_metric.GenAIClient",
                return_value=mock_gen_ai_client
                ):
            # Create test data
            metric_input = PerformanceInput(
                readme_text="This README mentions benchmarks but no metrics."
                )

            # Call the calculate method
            result = await self.metric.calculate(metric_input)

            # Assert the result
            expected_result = (
                PerformanceClaimsMetric.HAS_BENCHMARKS_WEIGHT * 0.5 +
                PerformanceClaimsMetric.HAS_METRICS_WEIGHT * 0.0
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_calculate_missing_keys(self):
        # Mock the GenAIClient with missing keys
        mock_gen_ai_client = AsyncMock()
        mock_gen_ai_client.get_performance_claims.return_value = {}

        # Patch the GenAIClient to use the mock
        with patch(
                "src.metrics.performance_claims_metric.GenAIClient",
                return_value=mock_gen_ai_client
                ):
            # Create test data
            metric_input = PerformanceInput(readme_text="Some README text")

            # Call the calculate method
            result = await self.metric.calculate(metric_input)

            # Assert the result (should default to 0 for missing keys)
            expected_result = 0.0
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_calculate_invalid_type(self):
        # Test with invalid input type
        with pytest.raises(AssertionError):
            await self.metric.calculate({"readme_text": "invalid"})
