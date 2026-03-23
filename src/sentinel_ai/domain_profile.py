"""Load domain profiles from built-in YAML or custom paths."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import yaml

from .models import DomainProfileConfig, SeverityThresholds

logger = logging.getLogger(__name__)

BUILTIN_PROFILES = ("defence-welfare", "general")


def load_profile(name_or_path: str) -> DomainProfileConfig:
    """Load a domain profile by built-in name or custom YAML path."""
    path = Path(name_or_path)
    if path.is_file():
        return _load_yaml(path)

    if name_or_path in BUILTIN_PROFILES:
        return _load_builtin(name_or_path)

    raise ValueError(
        f"Unknown domain profile: {name_or_path!r}. "
        f"Built-in profiles: {', '.join(BUILTIN_PROFILES)}. "
        f"Or provide a path to a custom YAML file."
    )


def _load_builtin(name: str) -> DomainProfileConfig:
    files = resources.files("sentinel_ai") / "profiles" / f"{name}.yml"
    text = files.read_text(encoding="utf-8")
    return _parse_yaml(text, source=f"built-in:{name}")


def _load_yaml(path: Path) -> DomainProfileConfig:
    text = path.read_text(encoding="utf-8")
    return _parse_yaml(text, source=str(path))


def _parse_yaml(text: str, source: str) -> DomainProfileConfig:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Domain profile must be a YAML mapping ({source})")

    # Convert severity_thresholds from nested dicts to SeverityThresholds
    raw_thresholds = data.get("severity_thresholds", {})
    parsed_thresholds: dict[str, SeverityThresholds] = {}
    for key, val in raw_thresholds.items():
        if isinstance(val, dict):
            parsed_thresholds[key] = SeverityThresholds(**val)
        else:
            parsed_thresholds[key] = SeverityThresholds()
    data["severity_thresholds"] = parsed_thresholds

    return DomainProfileConfig.model_validate(data)
