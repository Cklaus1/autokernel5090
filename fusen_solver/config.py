"""Configuration loading and defaults.

Loads from ~/.fusen_solver/config.yaml if it exists, otherwise uses defaults.
Supports environment variable overrides for API keys.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "~/.fusen_solver/config.yaml"


@dataclass
class Config:
    """Fusen Solver configuration."""

    backends: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    learning: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)


def default_config() -> Config:
    """Return sensible defaults (vLLM on localhost)."""
    return Config(
        backends={
            "primary": {
                "type": "vllm",
                "url": "http://localhost:8000/v1",
                "model": "default",
                "supports_batch": True,
                "max_context": 131072,
            },
        },
        strategy={
            "default_n": 4,
            "auto_n": True,
        },
        learning={
            "enabled": True,
            "db_path": "~/.fusen_solver/history.json",
            "min_data_for_adaptation": 10,
        },
        scoring={
            "test_weight": 0.4,
            "review_weight": 0.3,
            "diff_weight": 0.15,
            "syntax_weight": 0.1,
            "confidence_weight": 0.05,
        },
    )


def load_config(path: str | None = None) -> Config:
    """Load config from YAML file, falling back to defaults.

    Tries to import PyYAML; if unavailable, tries JSON format.
    Environment variables override file values for sensitive fields:
    - FUSEN_BACKEND_TYPE
    - FUSEN_BACKEND_URL
    - FUSEN_BACKEND_MODEL
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    """
    config = default_config()
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()

    if config_path.exists():
        try:
            raw = config_path.read_text()

            # Try YAML first
            try:
                import yaml
                data = yaml.safe_load(raw)
            except ImportError:
                # Fall back to JSON
                data = json.loads(raw)

            if isinstance(data, dict):
                if "backends" in data:
                    config.backends = data["backends"]
                if "strategy" in data:
                    config.strategy = data["strategy"]
                if "learning" in data:
                    config.learning = data["learning"]
                if "scoring" in data:
                    config.scoring = data["scoring"]

            logger.info("Loaded config from %s", config_path)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)

    # Environment variable overrides
    if os.environ.get("FUSEN_BACKEND_TYPE"):
        config.backends.setdefault("primary", {})["type"] = os.environ["FUSEN_BACKEND_TYPE"]
    if os.environ.get("FUSEN_BACKEND_URL"):
        config.backends.setdefault("primary", {})["url"] = os.environ["FUSEN_BACKEND_URL"]
    if os.environ.get("FUSEN_BACKEND_MODEL"):
        config.backends.setdefault("primary", {})["model"] = os.environ["FUSEN_BACKEND_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        config.backends.setdefault("primary", {}).setdefault("api_key", os.environ["OPENAI_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        config.backends.setdefault("primary", {}).setdefault("api_key", os.environ["ANTHROPIC_API_KEY"])

    return config


def save_default_config(path: str | None = None) -> None:
    """Write a default config file as a starting point."""
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    template = """\
# Fusen Solver Configuration
# See: https://github.com/autokernel/fusen-solver

backends:
  primary:
    type: vllm
    url: http://localhost:8000/v1
    model: default
    max_context: 131072

  # fallback:
  #   type: anthropic
  #   model: claude-sonnet-4-20250514
  #   max_context: 200000

strategy:
  default_n: 4
  auto_n: true

learning:
  enabled: true
  db_path: ~/.fusen_solver/history.json
  min_data_for_adaptation: 10

scoring:
  test_weight: 0.4
  review_weight: 0.3
  diff_weight: 0.15
  syntax_weight: 0.1
  confidence_weight: 0.05
"""
    config_path.write_text(template)
    logger.info("Default config written to %s", config_path)
