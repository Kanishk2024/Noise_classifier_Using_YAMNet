"""Helpers: paths, logging, config loading."""
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

DEFAULT_CONFIG: dict = {
    "audio": {"target_sr": 16000, "channels": 1},
    "classification": {
        "model_url": "https://tfhub.dev/google/yamnet/1",
        "class_map_url": (
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/audioset/yamnet/yamnet_class_map.csv"
        ),
        "window_sec": 1.0,
        "threshold": 0.15,
    },
    "noise": {"calibration_db": None},
    "output": {"plot": True, "keep_wav": False},
}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # keep stdout clean for the report
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (defaults filled in for anything missing)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        _deep_update(cfg, user_cfg)
    else:
        get_logger(__name__).warning("Config file %s not found; using defaults.", path)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
