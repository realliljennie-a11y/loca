# config.py — shared configuration loader

import os
import sys
from pathlib import Path
import yaml

# Config files live in the same directory as the scripts
_BASE = Path(__file__).parent

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

def load() -> dict:
    """Load config layers in order, deep-merging each on top of the previous.

    Load order:
      config.yaml          — required; AI/conversation settings
      audio_device.yaml    — optional; audio input/output hardware config
      ble_device.yaml      — optional; BLE device config
      hfp_secret.yaml      — optional; HA credentials, API keys, prompt overrides
    """
    cfg = _load_yaml(_BASE / "config.yaml")

    for name in ("audio_device.yaml", "ble_device.yaml"):
        path = _BASE / name
        if path.exists():
            cfg = _deep_merge(cfg, yaml.safe_load(path.read_text()) or {})

    secret_path = _BASE / "hfp_secret.yaml"
    if secret_path.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(secret_path.read_text()) or {})
    else:
        print("Warning: hfp_secret.yaml not found — HA integration and cloud LLMs disabled.",
              file=sys.stderr)

    cfg.setdefault("ha_url", "")
    cfg.setdefault("ha_token", "")

    return cfg

# Module-level singleton — import and use directly
CFG = load()
