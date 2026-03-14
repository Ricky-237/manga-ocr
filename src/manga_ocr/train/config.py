from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = {
    "data",
    "teachers",
    "synthetic",
    "pseudo_label",
    "annotation",
    "detector",
    "router",
    "recognizers",
    "distillation",
    "quantization",
    "benchmarks",
}


def load_train_config(config_path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the 'train' extra to load YAML training configs.") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    missing = REQUIRED_SECTIONS.difference(payload)
    if missing:
        raise ValueError(f"Missing required sections in {path}: {sorted(missing)}")
    return payload
