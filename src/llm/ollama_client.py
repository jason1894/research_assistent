"""Ollama API client for interacting with local LLM models."""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    """Client for the Ollama REST API.

    Provides methods for text generation, chat, model management, and
    returns a LangChain-compatible wrapper when needed.

    Args:
        base_url: Base URL of the Ollama server.
        model: Default model name to use.
        temperature: Sampling temperature (0-1).
        max_tokens: Maximum number of tokens to generate.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as exc:
            logger.warning("Ollama connection check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        """Return a list of models available on the Ollama server.

        Returns:
            A list of model metadata dicts.

        Raises:
            OllamaClientError: If the API call fails.
        """
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as exc:
            raise OllamaClientError(f"Failed to list models: {exc}") from exc

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama model registry.

        Args:
            model_name: Name of the model to pull (e.g. "llama2").

        Returns:
            True on success.

        Raises:
            OllamaClientError: If the pull request fails.
        """
        logger.info("Pulling model: %s", model_name)
        try:
            response = self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=600,
            )
            response.raise_for_status()
            logger.info("Successfully pulled model: %s", model_name)
            return True
        except requests.RequestException as exc:
            raise OllamaClientError(f"Failed to pull model '{model_name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str | Generator[str, None, None]:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            model: Model override; uses ``self.model`` if not provided.
            stream: If True, returns a generator yielding text chunks.
            options: Additional Ollama generation options.

        Returns:
            The generated text string, or a generator of chunks when
            ``stream=True``.

        Raises:
            OllamaClientError: If the API call fails.
        """
        effective_model = model or self.model
        payload: Dict[str, Any] = {
            "model": effective_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                **(options or {}),
            },
        }

        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=self.timeout,
            )
            response.raise_for_status()

            if stream:
                return self._stream_generate(response)

            result = response.json()
            return result.get("response", "")

        except requests.RequestException as exc:
            raise OllamaClientError(f"Generation failed: {exc}") from exc

    def _stream_generate(self, response: requests.Response) -> Generator[str, None, None]:
        """Yield text chunks from a streaming generate response."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if chunk := data.get("response", ""):
                        yield chunk
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str | Generator[str, None, None]:
        """Send a chat request to Ollama.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            model: Model override.
            stream: If True, returns a generator of text chunks.
            options: Additional Ollama generation options.

        Returns:
            The assistant message string, or a generator when ``stream=True``.

        Raises:
            OllamaClientError: If the API call fails.
        """
        effective_model = model or self.model
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                **(options or {}),
            },
        }

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=self.timeout,
            )
            response.raise_for_status()

            if stream:
                return self._stream_chat(response)

            result = response.json()
            return result.get("message", {}).get("content", "")

        except requests.RequestException as exc:
            raise OllamaClientError(f"Chat request failed: {exc}") from exc

    def _stream_chat(self, response: requests.Response) -> Generator[str, None, None]:
        """Yield content chunks from a streaming chat response."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if chunk := data.get("message", {}).get("content", ""):
                        yield chunk
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def get_langchain_llm(self, model: Optional[str] = None):
        """Return a LangChain-compatible Ollama LLM instance.

        Requires ``langchain-community`` to be installed.

        Args:
            model: Model override.

        Returns:
            A ``langchain_community.llms.Ollama`` instance.
        """
        try:
            from langchain_community.llms import Ollama  # type: ignore

            return Ollama(
                base_url=self.base_url,
                model=model or self.model,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for get_langchain_llm(). "
                "Install it with: pip install langchain-community"
            ) from exc
