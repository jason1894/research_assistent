"""Model manager for handling multiple LLM models and LoRA adapters."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .ollama_client import OllamaClient, OllamaClientError

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

    Integrates with :class:`~src.llm.ollama_client.OllamaClient` to list,
    load, switch, and apply adapters to models.

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
        self.client = OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model", "llama2"),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )
        self._current_model: str = llm_cfg.get("model", "llama2")
        self._loaded_adapter: Optional[str] = None
        self._available_models: List[Dict[str, Any]] = []

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
        """Return models available on the Ollama server.

        Returns:
            A list of model metadata dicts.
        """
        try:
            self._available_models = self.client.list_models()
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
        """Ensure a model is available on the Ollama server.

        Pulls the model if it is not already present.

        Args:
            name: Model name (e.g. "llama2", "mistral").

        Returns:
            True if the model is ready for use.
        """
        available = [m.get("name", "") for m in self.list_available_models()]
        if any(name in m for m in available):
            logger.info("Model '%s' is already available.", name)
            return True

        logger.info("Model '%s' not found locally – pulling...", name)
        try:
            return self.client.pull_model(name)
        except OllamaClientError as exc:
            logger.error("Failed to load model '%s': %s", name, exc)
            return False

    def switch_model(self, name: str) -> bool:
        """Switch the active model used for inference.

        Args:
            name: Name of the model to switch to.

        Returns:
            True if the switch was successful.
        """
        if self.load_model(name):
            self._current_model = name
            self.client.model = name
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
        return {
            "current_model": self._current_model,
            "loaded_adapter": self._loaded_adapter,
            "base_url": self.client.base_url,
            "temperature": self.client.temperature,
            "max_tokens": self.client.max_tokens,
            "server_reachable": self.client.check_connection(),
        }
