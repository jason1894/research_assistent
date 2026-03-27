"""LoRA training configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    All fields map directly to PEFT / Transformers / TRL training arguments.
    Sensible defaults are provided for each field.

    Args:
        r: LoRA rank – the dimension of the low-rank matrices.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability applied to LoRA layers.
        target_modules: Transformer module names to apply LoRA to.
        bias: Bias training strategy (``"none"``, ``"all"``, or
            ``"lora_only"``).
        task_type: PEFT task type (``"CAUSAL_LM"``, ``"SEQ_2_SEQ_LM"``…).
        base_model: HuggingFace model ID or local path.
        output_dir: Directory where the trained adapter is saved.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU/CPU.
        gradient_accumulation_steps: Number of steps before an optimiser
            update.
        learning_rate: Peak learning rate.
        max_seq_length: Maximum token sequence length.
        load_in_4bit: Enable 4-bit quantisation via bitsandbytes.
        load_in_8bit: Enable 8-bit quantisation via bitsandbytes.
        warmup_ratio: Fraction of total steps used for LR warmup.
        weight_decay: L2 regularisation weight decay.
        logging_steps: Log a training metric every N steps.
        save_steps: Save a checkpoint every N steps.
        eval_steps: Run evaluation every N steps (set to 0 to disable).
        data_path: Path to the training data file (JSON/JSONL/CSV).
        val_split: Fraction of training data used for validation.
        fp16: Enable FP16 mixed precision.
        bf16: Enable BF16 mixed precision (A100+).
    """

    # LoRA hyperparameters
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Model
    base_model: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./data/lora_adapters"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Quantisation
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Data
    data_path: Optional[str] = None
    val_split: float = 0.1

    # Precision
    fp16: bool = False
    bf16: bool = False

    def to_peft_config(self) -> "peft.LoraConfig":  # type: ignore[name-defined]
        """Convert to a PEFT ``LoraConfig`` instance.

        Raises:
            ImportError: If the ``peft`` package is not installed.
        """
        from peft import LoraConfig, TaskType  # type: ignore

        task_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_map.get(self.task_type, TaskType.CAUSAL_LM),
        )

    def to_training_arguments(self) -> "transformers.TrainingArguments":  # type: ignore[name-defined]
        """Convert to a Transformers ``TrainingArguments`` instance.

        Raises:
            ImportError: If the ``transformers`` package is not installed.
        """
        from transformers import TrainingArguments  # type: ignore

        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps if self.eval_steps > 0 else None,
            fp16=self.fp16,
            bf16=self.bf16,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoRAConfig":
        """Load a LoRAConfig from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A populated ``LoRAConfig`` instance.
        """
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        lora_section = data.get("lora", data)
        return cls(**{k: v for k, v in lora_section.items() if k in cls.__dataclass_fields__})
