"""Configuration loader for SAM3-RS.

Supports loading ``InferenceConfig`` from YAML or JSON files so that
experiments can be reproduced without editing Python code.

Supported file formats
----------------------
YAML example (recommended)::

    checkpoint_path: weights/sam3.pt
    bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz
    device: cuda
    confidence_threshold: 0.5
    prob_threshold: 0.1
    use_semantic_head: true
    use_instance_head: true
    use_presence_score: false
    slide_crop_size: 1024
    slide_stride: 512
    prompts_file: configs/loveda_classes.txt
    use_prompted_background: true

JSON example::

    {
        "checkpoint_path": "weights/sam3.pt",
        "bpe_path": "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "device": "cuda",
        "confidence_threshold": 0.5,
        "prob_threshold": 0.1,
        "prompts_file": "configs/loveda_classes.txt"
    }

Usage::

    from segmentor_lib.config_loader import load_inference_config
    config = load_inference_config("configs/experiment.yaml")
    segmentor = SAM3RSSegmentor(config)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import fields
from typing import Any, Dict, Optional


def load_inference_config(path: str, overrides: Optional[Dict[str, Any]] = None):
    """Load an ``InferenceConfig`` from a YAML or JSON file.

    Unknown keys in the file are silently ignored so that config files written
    for future versions remain backward-compatible.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` or ``.json`` config file.
        overrides: Optional dict of key-value pairs that overwrite values from
            the file (useful for command-line argument merging).

    Returns:
        An ``InferenceConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognised.
    """
    # Inline import to keep module importable without optional dependencies
    from segmentor import InferenceConfig  # noqa: PLC0415 – local import intentional

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        data = _load_yaml(path)
    elif ext == ".json":
        data = _load_json(path)
    else:
        raise ValueError(
            f"Unsupported config format '{ext}'. Use '.yaml', '.yml', or '.json'."
        )

    # Apply overrides
    if overrides:
        data.update(overrides)

    # Filter to only known InferenceConfig fields
    known = {f.name for f in fields(InferenceConfig)}
    filtered = {k: v for k, v in data.items() if k in known}

    unknown = set(data.keys()) - known
    if unknown:
        warnings.warn(
            f"[config_loader] Ignoring unknown config keys: {sorted(unknown)}",
            UserWarning,
            stacklevel=2,
        )

    return InferenceConfig(**filtered)


def save_inference_config(config, path: str) -> None:
    """Serialise an ``InferenceConfig`` to a YAML or JSON file.

    Args:
        config: An ``InferenceConfig`` instance.
        path: Destination file path.  Extension determines the format.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    from dataclasses import asdict  # noqa: PLC0415

    data = asdict(config)

    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if ext in (".yaml", ".yml"):
        _save_yaml(data, path)
    elif ext == ".json":
        _save_json(data, path)
    else:
        raise ValueError(
            f"Unsupported config format '{ext}'. Use '.yaml', '.yml', or '.json'."
        )

    print(f"✓ Config saved to {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs.  "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_yaml(data: Dict[str, Any], path: str) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to save YAML configs.  "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
