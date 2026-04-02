"""Unit tests for HuggingFaceClient."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.huggingface_client import HuggingFaceClient, HuggingFaceClientError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs) -> HuggingFaceClient:
    """Return a HuggingFaceClient with default test values."""
    defaults = dict(
        model_id="test-org/test-model",
        temperature=0.7,
        max_new_tokens=256,
        load_in_4bit=False,
    )
    defaults.update(kwargs)
    return HuggingFaceClient(**defaults)


# ---------------------------------------------------------------------------
# check_connection
# ---------------------------------------------------------------------------

class TestCheckConnection:
    def _mock_auto_config(self, side_effect=None, return_value=None):
        """Return a context manager that stubs AutoConfig in sys.modules."""
        mock_config = MagicMock()
        if side_effect is not None:
            mock_config.from_pretrained.side_effect = side_effect
        else:
            mock_config.from_pretrained.return_value = return_value or MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoConfig = mock_config
        return patch.dict("sys.modules", {"transformers": mock_transformers})

    def test_returns_true_when_local_cache_exists(self):
        client = _make_client()
        with self._mock_auto_config(return_value=MagicMock()):
            assert client.check_connection() is True

    def test_falls_back_to_hub_when_no_local_cache(self):
        client = _make_client()
        call_count = {"n": 0}

        def side_effect(model_id, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            call_count["n"] += 1
            return MagicMock()

        mock_config = MagicMock()
        mock_config.from_pretrained.side_effect = side_effect
        mock_transformers = MagicMock()
        mock_transformers.AutoConfig = mock_config

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = client.check_connection()

        assert result is True
        assert call_count["n"] == 1

    def test_returns_false_when_both_checks_fail(self):
        client = _make_client()
        with self._mock_auto_config(side_effect=OSError("unavailable")):
            assert client.check_connection() is False


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def _patched_client(self, generated_text: str = "test output") -> HuggingFaceClient:
        """Return a client with a pre-loaded mock pipeline."""
        client = _make_client()
        mock_pipeline = MagicMock(
            return_value=[{"generated_text": generated_text}]
        )
        mock_tokenizer = MagicMock()
        client._pipeline = mock_pipeline
        client._tokenizer = mock_tokenizer
        return client

    def test_returns_generated_text(self):
        client = self._patched_client("Hello from HF!")
        result = client.generate("Say hello")
        assert result == "Hello from HF!"

    def test_passes_options_to_pipeline(self):
        client = self._patched_client()
        client.generate("prompt", options={"repetition_penalty": 1.1})
        _, kwargs = client._pipeline.call_args
        assert kwargs.get("repetition_penalty") == 1.1

    def test_disables_sampling_when_temperature_zero(self):
        client = self._patched_client()
        client.temperature = 0.0
        client.generate("prompt")
        _, kwargs = client._pipeline.call_args
        assert kwargs["do_sample"] is False

    def test_raises_client_error_on_pipeline_exception(self):
        client = _make_client()
        client._pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))
        client._tokenizer = MagicMock()
        with pytest.raises(HuggingFaceClientError, match="Generation failed"):
            client.generate("prompt")

    def test_model_param_is_ignored(self):
        client = self._patched_client("ignored model test")
        result = client.generate("prompt", model="some-other-model")
        assert result == "ignored model test"


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

class TestChat:
    def _patched_client(self, generated_text: str = "chat response") -> HuggingFaceClient:
        client = _make_client()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted prompt>"
        mock_pipeline = MagicMock(return_value=[{"generated_text": generated_text}])
        client._pipeline = mock_pipeline
        client._tokenizer = mock_tokenizer
        return client

    def test_returns_assistant_response(self):
        client = self._patched_client("Paris")
        result = client.chat([{"role": "user", "content": "Capital of France?"}])
        assert result == "Paris"

    def test_applies_chat_template(self):
        client = self._patched_client()
        messages = [{"role": "user", "content": "Hello"}]
        client.chat(messages)
        client._tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )

    def test_falls_back_when_no_chat_template(self):
        client = _make_client()
        client._tokenizer = MagicMock()
        client._tokenizer.apply_chat_template.side_effect = Exception("no template")
        client._pipeline = MagicMock(return_value=[{"generated_text": "fallback"}])

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = client.chat(messages)
        assert result == "fallback"

        # Ensure the pipeline was called with a fallback prompt
        args, _ = client._pipeline.call_args
        prompt = args[0]
        assert "System: You are helpful." in prompt
        assert "User: Hi" in prompt
        assert "Assistant:" in prompt


# ---------------------------------------------------------------------------
# _ensure_loaded — import error handling
# ---------------------------------------------------------------------------

class TestEnsureLoaded:
    def test_raises_client_error_on_import_error(self):
        client = _make_client()
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with pytest.raises(HuggingFaceClientError):
                client._ensure_loaded()


# ---------------------------------------------------------------------------
# get_langchain_llm
# ---------------------------------------------------------------------------

class TestGetLangchainLlm:
    def test_returns_huggingface_pipeline_wrapper(self):
        client = _make_client()
        client._pipeline = MagicMock()
        client._tokenizer = MagicMock()

        mock_hf_pipeline_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()},
        ):
            with patch(
                "langchain_community.llms.HuggingFacePipeline",
                mock_hf_pipeline_cls,
            ):
                result = client.get_langchain_llm()

        mock_hf_pipeline_cls.assert_called_once_with(pipeline=client._pipeline)
        assert result is mock_hf_pipeline_cls.return_value
