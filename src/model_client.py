"""Unified model client routing all calls through the yunwu OpenAI-compatible API."""

import os
import time
import logging
from pathlib import Path

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
_client: OpenAI | None = None
_model_registry: dict[str, dict] = {}


def _load_config() -> None:
    global _client, _model_registry
    if _client is not None:
        return

    api_key = os.environ.get("YUNWU_API_KEY")
    if not api_key:
        raise EnvironmentError("YUNWU_API_KEY environment variable is not set.")

    _client = OpenAI(
        api_key=api_key,
        base_url="https://yunwu.ai/v1",
    )

    config = yaml.safe_load(_CONFIG_PATH.read_text())
    for entry in config.get("eval_models", []):
        _model_registry[entry["name"]] = entry
    judge = config.get("judge_model")
    if judge:
        _model_registry[judge["name"]] = judge


def _call_with_retry(
    model_id: str,
    prompt: str,
    temperature: float,
    max_retries: int = 5,
    initial_wait: float = 2.0,
) -> str:
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed ({e}), retrying in {wait}s…")
            time.sleep(wait)
            wait = min(wait * 2, 60)
    return ""


def call_model(model_name: str, prompt: str, temperature: float = 0.0) -> str:
    """Call a model by its config name and return the text response."""
    _load_config()
    entry = _model_registry.get(model_name)
    if entry is None:
        raise ValueError(f"Model '{model_name}' not found in models.yaml.")
    return _call_with_retry(entry["model_id"], prompt, temperature)


def list_models() -> list[str]:
    _load_config()
    return [name for name in _model_registry if name != "gpt-5.5"]


def get_judge_model_name() -> str:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    return config["judge_model"]["name"]
