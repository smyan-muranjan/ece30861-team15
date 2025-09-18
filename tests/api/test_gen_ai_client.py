import os
import pytest
from unittest.mock import patch, MagicMock
from src.api.gen_ai_client import GenAIClient

class TestGenAIClient:
    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    def test_init_sets_headers(self):
        client = GenAIClient()
        assert client.headers["Authorization"] == "Bearer test_key"
        assert client.headers["Content-Type"] == "application/json"
        assert client.url == "https://genai.rcac.purdue.edu/api/chat/completions"

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("src.api.gen_ai_client.requests.post")
    def test_chat_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Hello, world!"}}
            ]
        }
        mock_post.return_value = mock_response
        client = GenAIClient()
        result = client.chat("Hi")
        assert result == "Hello, world!"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == client.headers
        assert kwargs["json"]["messages"][0]["content"] == "Hi"

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("src.api.gen_ai_client.requests.post")
    def test_chat_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        client = GenAIClient()
        with pytest.raises(Exception) as exc:
            client.chat("Hi")
        assert "Error: 400" in str(exc.value)
        assert "Bad Request" in str(exc.value)

    @patch.dict(os.environ, {"GENAI_API_KEY": "test_key"})
    @patch("src.api.gen_ai_client.requests.post")
    def test_chat_custom_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Model response"}}
            ]
        }
        mock_post.return_value = mock_response
        client = GenAIClient()
        result = client.chat("Test", model="custom-model")
        assert result == "Model response"
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "custom-model"
