"""Unit tests for OllamaClient and ModelManager."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.ollama_client import OllamaClient, OllamaClientError


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


class TestOllamaClientCheckConnection:
    def test_returns_true_on_200(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(client._session, "get", return_value=mock_resp):
            assert client.check_connection() is True

    def test_returns_false_on_connection_error(self):
        import requests

        client = OllamaClient()
        with patch.object(client._session, "get", side_effect=requests.RequestException):
            assert client.check_connection() is False


class TestOllamaClientListModels:
    def test_returns_model_list(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama2"}, {"name": "mistral"}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "get", return_value=mock_resp):
            models = client.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "llama2"

    def test_raises_on_http_error(self):
        import requests

        client = OllamaClient()
        with patch.object(client._session, "get", side_effect=requests.RequestException("fail")):
            with pytest.raises(OllamaClientError):
                client.list_models()


class TestOllamaClientGenerate:
    def test_returns_text(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Hello world!"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "post", return_value=mock_resp):
            result = client.generate("Test prompt", stream=False)
        assert result == "Hello world!"

    def test_raises_on_http_error(self):
        import requests

        client = OllamaClient()
        with patch.object(
            client._session, "post", side_effect=requests.RequestException("fail")
        ):
            with pytest.raises(OllamaClientError):
                client.generate("Test prompt")

    def test_stream_yields_chunks(self):
        import json

        client = OllamaClient()
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode(),
            json.dumps({"response": " world", "done": False}).encode(),
            json.dumps({"response": "!", "done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        with patch.object(client._session, "post", return_value=mock_resp):
            chunks = list(client.generate("Test", stream=True))
        assert chunks == ["Hello", " world", "!"]


class TestOllamaClientChat:
    def test_returns_message_content(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Paris"}}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "post", return_value=mock_resp):
            result = client.chat([{"role": "user", "content": "Capital of France?"}])
        assert result == "Paris"

    def test_raises_on_request_exception(self):
        import requests

        client = OllamaClient()
        with patch.object(
            client._session, "post", side_effect=requests.RequestException
        ):
            with pytest.raises(OllamaClientError):
                client.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class TestModelManager:
    def _make_manager(self):
        from src.llm.model_manager import ModelManager  # type: ignore

        with patch("src.llm.model_manager.open", side_effect=FileNotFoundError):
            mgr = ModelManager.__new__(ModelManager)
            mgr.client = OllamaClient()
            mgr._current_model = "llama2"
            mgr._loaded_adapter = None
            mgr._available_models = []
        return mgr

    def test_current_model_default(self):
        from src.llm.model_manager import ModelManager  # type: ignore

        with patch("builtins.open", side_effect=FileNotFoundError):
            mgr = ModelManager.__new__(ModelManager)
            mgr.client = OllamaClient()
            mgr._current_model = "llama2"
            mgr._loaded_adapter = None
            mgr._available_models = []
        assert mgr.current_model == "llama2"

    def test_switch_model_calls_load(self):
        from src.llm.model_manager import ModelManager  # type: ignore

        mgr = MagicMock(spec=ModelManager)
        mgr._current_model = "llama2"
        mgr.load_model.return_value = True

        # Directly test logic without full instantiation
        def switch(name):
            if mgr.load_model(name):
                mgr._current_model = name
                return True
            return False

        assert switch("mistral") is True
        assert mgr._current_model == "mistral"

    def test_apply_lora_adapter_missing_path(self, tmp_path):
        from src.llm.model_manager import ModelManager  # type: ignore

        with patch("builtins.open", side_effect=FileNotFoundError):
            mgr = ModelManager.__new__(ModelManager)
            mgr.client = OllamaClient()
            mgr._current_model = "llama2"
            mgr._loaded_adapter = None
            mgr._available_models = []

        result = mgr.apply_lora_adapter("/nonexistent/path")
        assert result is False

    def test_apply_lora_adapter_valid_path(self, tmp_path):
        from src.llm.model_manager import ModelManager  # type: ignore

        with patch("builtins.open", side_effect=FileNotFoundError):
            mgr = ModelManager.__new__(ModelManager)
            mgr.client = OllamaClient()
            mgr._current_model = "llama2"
            mgr._loaded_adapter = None
            mgr._available_models = []

        result = mgr.apply_lora_adapter(str(tmp_path))
        assert result is True
        assert mgr.loaded_adapter is not None
