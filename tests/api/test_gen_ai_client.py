import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.gen_ai_client import GenAIClient


class TestGenAIClient:
    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    def test_init_sets_headers(self):
        client = GenAIClient()
        assert client.headers["Authorization"] == "Bearer test_key"
        assert client.headers["Content-Type"] == "application/json"
        assert client.url == (
            "https://genai.rcac.purdue.edu/api/chat/completions"
        )

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("aiohttp.ClientSession.post")
    @pytest.mark.asyncio
    async def test_chat_success(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {"message": {"content": "Hello, world!"}}
            ]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        client = GenAIClient()
        result = await client.chat("Hi")
        assert result == "Hello, world!"
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("aiohttp.ClientSession.post")
    @pytest.mark.asyncio
    async def test_chat_error(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_post.return_value.__aenter__.return_value = mock_response

        client = GenAIClient()
        with pytest.raises(Exception) as exc:
            await client.chat("Hi")
        assert "Error: 400" in str(exc.value)
        assert "Bad Request" in str(exc.value)

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("aiohttp.ClientSession.post")
    @pytest.mark.asyncio
    async def test_chat_custom_model(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {"message": {"content": "Model response"}}
            ]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        client = GenAIClient()
        result = await client.chat("Test", model="custom-model")
        assert result == "Model response"

        # Verify the call was made with custom model
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "custom-model"

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("aiohttp.ClientSession.post")
    @patch("builtins.open", create=True)
    @pytest.mark.asyncio
    async def test_get_performance_claims(self, mock_open, mock_post):
        # Mock file reading
        mock_file = MagicMock()
        mock_file.read.return_value = "Test prompt: "
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock HTTP response - return valid JSON string that will be parsed
        expected_dict = {
            "mentions_benchmarks": 0.8,
            "has_metrics": 0.6
        }
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {"message": {"content": json.dumps(expected_dict)}}
            ]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        client = GenAIClient()
        result = await client.get_performance_claims("README content")

        # Verify the result is the expected dict
        assert result == expected_dict
        assert isinstance(result, dict)

        # Verify file was opened correctly
        mock_open.assert_called_once_with(
            "src/api/performance_claims_ai_prompt.txt", "r"
        )

        # Verify HTTP call was made
        mock_post.assert_called_once()
