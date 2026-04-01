"""Model manager for handling multiple LLM models and LoRA adapters."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .ollama_client import OllamaClient, OllamaClientError
from .huggingface_client import HuggingFaceClient, HuggingFaceClientError

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        logger.warning("config.yaml not found; using defaults.")
        return {}


class ModelManager:
    """Manages multiple LLM models and optional LoRA adapters.

    Supports two backends selected via ``config.yaml``:

    * ``"ollama"`` (default) – delegates to
      :class:`~src.llm.ollama_client.OllamaClient`.
    * ``"huggingface"`` – delegates to
      :class:`~src.llm.huggingface_client.HuggingFaceClient` for direct
      local inference using the HuggingFace Transformers library.

    Args:
        config_path: Path to ``config.yaml``.  Defaults to the standard
            project config location.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        cfg_path = config_path or _CONFIG_PATH
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        except FileNotFoundError:
            config = {}

        llm_cfg = config.get("llm", {})
        hf_cfg = config.get("huggingface", {})

        self._backend: str = llm_cfg.get("backend", "ollama")

        if self._backend == "huggingface":
            self.client: Union[OllamaClient, HuggingFaceClient] = HuggingFaceClient(
                model_id=hf_cfg.get(
                    "model_id",
                    "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
                ),
                temperature=hf_cfg.get("temperature", llm_cfg.get("temperature", 0.7)),
                max_new_tokens=hf_cfg.get("max_new_tokens", llm_cfg.get("max_tokens", 2048)),
                load_in_4bit=hf_cfg.get("load_in_4bit", True),
                device_map=hf_cfg.get("device_map", "auto"),
                torch_dtype=hf_cfg.get("torch_dtype", "auto"),
                trust_remote_code=hf_cfg.get("trust_remote_code", True),
            )
            self._current_model: str = hf_cfg.get(
                "model_id",
                "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
            )
        else:
            self.client = OllamaClient(
                base_url=llm_cfg.get("base_url", "http://localhost:11434"),
                model=llm_cfg.get("model", "llama2"),
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_tokens", 2048),
            )
            self._current_model = llm_cfg.get("model", "llama2")

        self._loaded_adapter: Optional[str] = None
        self._available_models: List[Dict[str, Any]] = []

    @property
    def backend(self) -> str:
        """Active backend identifier (``"ollama"`` or ``"huggingface"``)."""
        return self._backend

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        """Name of the currently active model."""
        return self._current_model

    @property
    def loaded_adapter(self) -> Optional[str]:
        """Path of the currently loaded LoRA adapter, or None."""
        return self._loaded_adapter

    # ------------------------------------------------------------------
    # Model operations
    # ------------------------------------------------------------------

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Return models available for the active backend.

        For the ``ollama`` backend this queries the Ollama server.
        For the ``huggingface`` backend a single-entry list describing
        the configured model is returned.

        Returns:
            A list of model metadata dicts.
        """
        if self._backend == "huggingface":
            hf_client = self.client  # type: ignore[assignment]
            self._available_models = [{"name": hf_client.model_id, "backend": "huggingface"}]
            return self._available_models
        try:
            self._available_models = self.client.list_models()  # type: ignore[union-attr]
            return self._available_models
        except OllamaClientError as exc:
            logger.error("Could not list models: %s", exc)
            return []

    def get_model(self, name: Optional[str] = None) -> str:
        """Return the effective model name.

        Args:
            name: Override model name.  If None, returns ``current_model``.
        """
        return name or self._current_model

    def load_model(self, name: str) -> bool:
        """Ensure a model is available.

        For the ``ollama`` backend, pulls the model if it is not already
        present on the server.  For the ``huggingface`` backend, verifies
        that the model can be reached (locally cached or on the Hub).

        Args:
            name: Ollama model name or HuggingFace model ID.

        Returns:
            True if the model is ready for use.
        """
        if self._backend == "huggingface":
            hf_client = self.client  # type: ignore[assignment]
            hf_client.model_id = name
            available = hf_client.check_connection()
            if not available:
                logger.error("HuggingFace model '%s' is not reachable.", name)
            return available

        available = [m.get("name", "") for m in self.list_available_models()]
        if any(name in m for m in available):
            logger.info("Model '%s' is already available.", name)
            return True

        logger.info("Model '%s' not found locally – pulling...", name)
        try:
            return self.client.pull_model(name)  # type: ignore[union-attr]
        except OllamaClientError as exc:
            logger.error("Failed to load model '%s': %s", name, exc)
            return False

    def switch_model(self, name: str) -> bool:
        """Switch the active model used for inference.

        For the ``ollama`` backend the server model is changed.  For the
        ``huggingface`` backend a new
        :class:`~src.llm.huggingface_client.HuggingFaceClient` is created
        with the new model ID (the previous model is unloaded).

        Args:
            name: Ollama model name or HuggingFace model ID.

        Returns:
            True if the switch was successful.
        """
        if self._backend == "huggingface":
            hf_client = self.client  # type: ignore[assignment]
            new_client = HuggingFaceClient(
                model_id=name,
                temperature=hf_client.temperature,
                max_new_tokens=hf_client.max_new_tokens,
                load_in_4bit=hf_client.load_in_4bit,
                device_map=hf_client.device_map,
                torch_dtype=hf_client.torch_dtype,
                trust_remote_code=hf_client.trust_remote_code,
            )
            if not new_client.check_connection():
                logger.error("HuggingFace model '%s' is not reachable.", name)
                return False
            self.client = new_client
            self._current_model = name
            self._loaded_adapter = None
            logger.info("Switched HuggingFace model to: %s", name)
            return True

        if self.load_model(name):
            self._current_model = name
            self.client.model = name  # type: ignore[union-attr]
            self._loaded_adapter = None  # adapters are model-specific
            logger.info("Switched to model: %s", name)
            return True
        return False

    # ------------------------------------------------------------------
    # LoRA adapter support
    # ------------------------------------------------------------------

    def apply_lora_adapter(self, adapter_path: str) -> bool:
        """Apply a LoRA adapter to the current model.

        This registers the adapter so that downstream components can
        incorporate it when building model pipelines.  Actual adapter
        merging at inference time requires the transformers/PEFT stack.

        Args:
            adapter_path: Path to the adapter directory produced by
                :class:`~src.lora.trainer.LoRATrainer`.

        Returns:
            True if the adapter path exists and was registered.
        """
        path = Path(adapter_path)
        if not path.exists():
            logger.error("Adapter path does not exist: %s", adapter_path)
            return False

        self._loaded_adapter = str(path.resolve())
        logger.info("LoRA adapter registered: %s", self._loaded_adapter)
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary of the current model state."""
        if self._backend == "huggingface":
            hf = self.client  # type: ignore[assignment]
            return {
                "backend": "huggingface",
                "current_model": self._current_model,
                "loaded_adapter": self._loaded_adapter,
                "model_id": hf.model_id,
                "temperature": hf.temperature,
                "max_new_tokens": hf.max_new_tokens,
                "load_in_4bit": hf.load_in_4bit,
                "model_available": hf.check_connection(),
            }
        return {
            "backend": "ollama",
            "current_model": self._current_model,
            "loaded_adapter": self._loaded_adapter,
            "base_url": self.client.base_url,  # type: ignore[union-attr]
            "temperature": self.client.temperature,
            "max_tokens": self.client.max_tokens,  # type: ignore[union-attr]
            "server_reachable": self.client.check_connection(),
        }
