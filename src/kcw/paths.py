"""Resolve Drive / repo paths from env or paths.yaml (never hardcode G:\\ alone)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _expand(path: str | Path) -> Path:
    """Expand user/~ only. Do NOT Path.resolve() Drive paths — on Windows,
    resolve() can rewrite G:\\Shared drives\\... into DriveFS cache under
    AppData, and large CSV overwrites then fail to sync (sidet/icmas symptom).
    """
    return Path(path).expanduser()


def repo_root() -> Path:
    env = os.getenv("KCW_ANALYTICS_ROOT")
    if env:
        return _expand(env)
    # src/kcw/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_paths_yaml() -> dict[str, Any]:
    candidates = [
        _expand(os.getenv("KCW_PATHS_YAML", "")),
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
        return _expand(env)

    cfg = _load_paths_yaml()
    if cfg.get("drive_root"):
        return _expand(cfg["drive_root"])

    # Common local fallbacks (Windows Drive File Stream / Colab)
    for candidate in (
        Path(r"G:\Shared drives"),
        Path("/content/drive/Shareddrives"),
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot resolve Drive root. Set KCW_DRIVE_ROOT or create paths.yaml "
        "with drive_root: ..."
    )


def analytics_root() -> Path:
    env = os.getenv("KCW_ANALYTICS_DATA_ROOT")
    if env:
        return _expand(env)

    cfg = _load_paths_yaml()
    if cfg.get("analytics_root"):
        return _expand(cfg["analytics_root"])

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
        return _expand(env)
    return repo_root() / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
