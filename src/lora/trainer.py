"""LoRA fine-tuning trainer using HuggingFace PEFT."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LoRAConfig

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Fine-tunes a causal language model using LoRA (via PEFT).

    Uses 4-bit quantisation by default for memory-efficient training on
    consumer hardware.

    Args:
        config: A :class:`~src.lora.config.LoRAConfig` instance.  If not
            provided, defaults are used.
    """

    def __init__(self, config: Optional[LoRAConfig] = None) -> None:
        self.config = config or LoRAConfig()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._dataset: Optional[Any] = None

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def prepare_model(self) -> None:
        """Load the base model and apply LoRA adapters.

        Uses 4-bit or 8-bit quantisation when configured.  After calling
        this method ``self._model`` and ``self._tokenizer`` are set.

        Raises:
            ImportError: If transformers/peft/bitsandbytes are not installed.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore

        logger.info("Loading base model: %s", self.config.base_model)

        bnb_config: Optional[Any] = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.config.load_in_4bit or self.config.load_in_8bit:
            self._model = prepare_model_for_kbit_training(self._model)

        peft_config = self.config.to_peft_config()
        self._model = get_peft_model(self._model, peft_config)
        self._model.print_trainable_parameters()
        logger.info("Model prepared with LoRA adapters.")

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, data_path: str) -> None:
        """Load and tokenise the training dataset.

        The dataset file must be JSON or JSONL with ``instruction`` and
        ``response`` fields (or ``input``/``output`` fields).

        Args:
            data_path: Path to the training data file.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ImportError: If the ``datasets`` package is not installed.
        """
        from datasets import load_dataset  # type: ignore

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        logger.info("Loading dataset from: %s", data_path)

        if path.suffix in {".json", ".jsonl"}:
            raw = load_dataset("json", data_files=str(path))
        elif path.suffix == ".csv":
            raw = load_dataset("csv", data_files=str(path))
        else:
            raise ValueError(f"Unsupported data format: {path.suffix}")

        def _format(example: Dict[str, Any]) -> Dict[str, Any]:
            instruction = example.get("instruction", example.get("input", ""))
            response = example.get("response", example.get("output", ""))
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return {"text": text}

        formatted = raw.map(_format)

        def _tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = formatted.map(_tokenize, batched=True, remove_columns=formatted["train"].column_names)

        split = tokenized["train"].train_test_split(test_size=self.config.val_split, seed=42)
        self._dataset = split
        logger.info(
            "Dataset prepared: %d train / %d eval samples.",
            len(split["train"]),
            len(split["test"]),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the LoRA fine-tuning loop.

        ``prepare_model()`` and ``prepare_dataset()`` must be called first.

        Raises:
            RuntimeError: If model or dataset have not been prepared.
        """
        if self._model is None:
            raise RuntimeError("Call prepare_model() before train().")
        if self._dataset is None:
            raise RuntimeError("Call prepare_dataset() before train().")

        from transformers import DataCollatorForLanguageModeling, Trainer  # type: ignore

        training_args = self.config.to_training_arguments()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset["train"],
            eval_dataset=self._dataset["test"],
            data_collator=data_collator,
        )

        logger.info("Starting LoRA fine-tuning …")
        trainer.train()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Adapter persistence
    # ------------------------------------------------------------------

    def save_adapter(self, path: Optional[str] = None) -> str:
        """Save the trained LoRA adapter weights to disk.

        Args:
            path: Output directory.  Defaults to ``config.output_dir``.

        Returns:
            The absolute path to the saved adapter directory.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("No model to save.  Call prepare_model() first.")

        output_path = Path(path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(output_path))
        self._tokenizer.save_pretrained(str(output_path))
        logger.info("Adapter saved to: %s", output_path)
        return str(output_path.resolve())

    def load_adapter(self, path: str) -> None:
        """Load a previously saved LoRA adapter.

        Args:
            path: Path to the adapter directory saved by :meth:`save_adapter`.

        Raises:
            FileNotFoundError: If the adapter directory does not exist.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import PeftModel  # type: ignore

        adapter_path = Path(path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {path}")

        logger.info("Loading adapter from: %s", path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, device_map="auto", trust_remote_code=True
        )
        self._model = PeftModel.from_pretrained(base, str(adapter_path))
        logger.info("Adapter loaded successfully.")

    def merge_and_save(self, output_path: Optional[str] = None) -> str:
        """Merge LoRA weights into the base model and save.

        The merged model can be used for inference without the PEFT library.

        Args:
            output_path: Directory for the merged model.  Defaults to
                ``config.output_dir + "_merged"``.

        Returns:
            The absolute path to the merged model directory.
        """
        if self._model is None:
            raise RuntimeError("No model available.  Call prepare_model() or load_adapter().")

        merged_path = Path(output_path or (str(self.config.output_dir) + "_merged"))
        merged_path.mkdir(parents=True, exist_ok=True)

        merged = self._model.merge_and_unload()
        merged.save_pretrained(str(merged_path))
        self._tokenizer.save_pretrained(str(merged_path))
        logger.info("Merged model saved to: %s", merged_path)
        return str(merged_path.resolve())
