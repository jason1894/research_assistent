"""Centralised logging configuration for the research assistant."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
_CONFIGURED = False


def _load_log_config() -> dict:
    """Read the logging section from config.yaml."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cfg.get("logging", {})
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def _configure_root_logger() -> None:
    """Set up root-level handlers (called once)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_cfg = _load_log_config()
    level_name: str = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = log_cfg.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    # File handler (optional)
    log_file: Optional[str] = log_cfg.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes: int = log_cfg.get("max_bytes", 10 * 1024 * 1024)
        backup_count: int = log_cfg.get("backup_count", 5)
        fh = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root handlers are configured.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
