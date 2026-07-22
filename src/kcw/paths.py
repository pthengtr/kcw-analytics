"""Resolve Drive / repo paths from env or paths.yaml (never hardcode G:\\ alone)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def repo_root() -> Path:
    env = os.getenv("KCW_ANALYTICS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/kcw/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_paths_yaml() -> dict[str, Any]:
    candidates = [
        Path(os.getenv("KCW_PATHS_YAML", "")),
        repo_root() / "paths.yaml",
        Path.home() / ".kcw" / "paths.yaml",
    ]
    for path in candidates:
        if not path or str(path) == ".":
            continue
        if path.is_file():
            if yaml is None:
                raise RuntimeError(
                    f"Found {path} but PyYAML is not installed. "
                    "pip install pyyaml or set KCW_DRIVE_ROOT instead."
                )
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"paths.yaml must be a mapping: {path}")
            return data
    return {}


def drive_root() -> Path:
    """
    Root folder that contains KCW-Data (or is already the analytics tree).

    Preferred env: KCW_DRIVE_ROOT
      e.g. G:\\Shared drives
           /content/drive/Shareddrives
           /mnt/gdrive

    Optional paths.yaml keys:
      drive_root: ...
      analytics_root: ...   # full path to kcw_analytics (overrides composition)
    """
    env = os.getenv("KCW_DRIVE_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    cfg = _load_paths_yaml()
    if cfg.get("analytics_root"):
        # parent of kcw_analytics/... is not always drive_root; expose via analytics_root()
        pass
    if cfg.get("drive_root"):
        return Path(cfg["drive_root"]).expanduser().resolve()

    # Common local fallbacks (Windows Drive File Stream / Colab)
    for candidate in (
        Path(r"G:\Shared drives"),
        Path("/content/drive/Shareddrives"),
    ):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot resolve Drive root. Set KCW_DRIVE_ROOT or create paths.yaml "
        "with drive_root: ..."
    )


def analytics_root() -> Path:
    env = os.getenv("KCW_ANALYTICS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    cfg = _load_paths_yaml()
    if cfg.get("analytics_root"):
        return Path(cfg["analytics_root"]).expanduser().resolve()

    return drive_root() / "KCW-Data" / "kcw_analytics"


def raw_dir() -> Path:
    return analytics_root() / "01_raw"


def curated_dir() -> Path:
    return analytics_root() / "03_curated"


def outputs_dir() -> Path:
    return analytics_root() / "04_outputs"


def tar_output_dir(kind: str = "TAR") -> Path:
    """kind: TAR | 3TAR"""
    return outputs_dir() / "06_TAR" / kind


def log_dir() -> Path:
    env = os.getenv("KCW_ANALYTICS_LOG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
