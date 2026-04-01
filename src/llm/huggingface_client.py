"""HuggingFace Transformers client for local LLM inference."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HuggingFaceClientError(Exception):
    """Raised when a HuggingFace model operation fails."""


class HuggingFaceClient:
    """Client for local LLM inference using HuggingFace Transformers.

    Loads a model from the HuggingFace Hub (or a local path) and exposes
    the same ``generate`` / ``chat`` / ``get_langchain_llm`` interface as
    :class:`~src.llm.ollama_client.OllamaClient`.

    The model is loaded lazily on first use so that importing this module is
    cheap even when ``torch`` / ``transformers`` are not installed.

    Args:
        model_id: HuggingFace model ID or local directory path.
        temperature: Sampling temperature (0 disables sampling).
        max_new_tokens: Maximum number of new tokens to generate.
        load_in_4bit: Load the model in 4-bit quantization via
            ``bitsandbytes`` (reduces VRAM requirements significantly).
        device_map: Device mapping strategy passed to
            ``AutoModelForCausalLM.from_pretrained`` (e.g. ``"auto"``).
        torch_dtype: Torch dtype string (``"auto"``, ``"float16"``,
            ``"bfloat16"``).  ``"auto"`` lets Transformers decide.
        trust_remote_code: Whether to trust and execute remote code
            included in the model repository.
    """

    def __init__(
        self,
        model_id: str = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        # Populated lazily by _ensure_loaded()
        self._pipeline = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model and tokenizer on first use."""
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )

            logger.info("Loading tokenizer: %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )

            model_kwargs: Dict[str, Any] = {
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
            }

            if self.torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype, torch.float16)
            else:
                model_kwargs["torch_dtype"] = "auto"

            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config

            logger.info("Loading model: %s", self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
            )
            logger.info("Model loaded successfully: %s", self.model_id)

        except ImportError as exc:
            raise HuggingFaceClientError(
                "transformers, torch and bitsandbytes are required for "
                "HuggingFaceClient.  Install them with:\n"
                "  pip install transformers torch bitsandbytes"
            ) from exc
        except Exception as exc:
            raise HuggingFaceClientError(
                f"Failed to load model '{self.model_id}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return True if the model is available locally or on the Hub.

        Checks for locally cached files first to avoid a network round-trip.
        Falls back to fetching the remote config if no local cache is found.
        """
        from transformers import AutoConfig  # noqa: PLC0415

        try:
            # Fast path: local cache
            AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
            return True
        except Exception:
            pass

        try:
            AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            return True
        except Exception as exc:
            logger.warning("HuggingFace model check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,  # ignored; model is set at init
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt string.
            model: Ignored – the model is fixed at initialisation time.
            stream: Streaming is not yet supported; always returns the full
                generated text.
            options: Additional keyword arguments forwarded to the
                ``transformers`` pipeline call.

        Returns:
            Generated text (excluding the input prompt).

        Raises:
            HuggingFaceClientError: If generation fails.
        """
        self._ensure_loaded()
        opts = options or {}
        try:
            do_sample = self.temperature > 0
            results = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if do_sample else 1.0,
                do_sample=do_sample,
                return_full_text=False,
                **opts,
            )
            return results[0]["generated_text"]
        except Exception as exc:
            raise HuggingFaceClientError(f"Generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,  # ignored
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run a multi-turn chat completion.

        Applies the model's chat template to convert the message list into a
        single prompt string, then calls :meth:`generate`.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts
                following the OpenAI message format.
            model: Ignored.
            stream: Not supported; always returns the full response.
            options: Additional generation kwargs.

        Returns:
            Assistant response string.

        Raises:
            HuggingFaceClientError: If the request fails.
        """
        self._ensure_loaded()
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without a chat template
            prompt = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages
            )
            prompt += "\nAssistant:"

        return self.generate(prompt, options=options)

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def get_langchain_llm(self, model: Optional[str] = None):
        """Return a LangChain-compatible ``HuggingFacePipeline`` LLM.

        Requires ``langchain-community`` and ``transformers`` to be installed.

        Args:
            model: Ignored.

        Returns:
            A ``langchain_community.llms.HuggingFacePipeline`` instance
            wrapping the loaded pipeline.
        """
        self._ensure_loaded()
        try:
            from langchain_community.llms import HuggingFacePipeline  # type: ignore

            return HuggingFacePipeline(pipeline=self._pipeline)
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for get_langchain_llm(). "
                "Install it with: pip install langchain-community"
            ) from exc
