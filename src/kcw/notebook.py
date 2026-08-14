"""Notebook bootstrap: Windows Drive, Colab, or Linux rclone.

Notebooks should not hardcode G:\\. Call setup() from the first cell:

    from src.kcw.notebook import setup
    cfg = setup()
    BASE_FOLDER = cfg.BASE_FOLDER
    BASE_FOLDER_GIT = cfg.BASE_FOLDER_GIT
    REPO_ROOT = cfg.REPO_ROOT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from .paths import analytics_root, drive_root, repo_root as _pkg_repo_root


def find_repo_root() -> Path:
    env = os.getenv("KCW_ANALYTICS_ROOT")
    if env:
        return Path(env).expanduser()
    for cand in (Path.cwd(), *Path.cwd().parents):
        if (cand / "src" / "kcw").is_dir():
            return cand
    colab = Path("/content/kcw-analytics")
    if (colab / "src" / "kcw").is_dir():
        return colab
    return _pkg_repo_root()


def put_repo_on_path() -> Path:
    root = find_repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def setup(*, mount_colab: bool = True) -> SimpleNamespace:
    """Resolve Drive + repo paths. Safe on Windows, Linux, and Colab.

    Sets KCW_DRIVE_ROOT on Colab after mounting Shareddrives.
    """
    if mount_colab and "google.colab" in sys.modules:
        os.environ.setdefault("KCW_DRIVE_ROOT", "/content/drive/Shareddrives")
        try:
            from google.colab import drive

            if not Path("/content/drive/Shareddrives").exists():
                drive.mount("/content/drive")
        except Exception:
            pass

    root = put_repo_on_path()
    base = drive_root()
    analytics = analytics_root()
    return SimpleNamespace(
        BASE_FOLDER=str(base),
        BASE_FOLDER_GIT=str(root.parent),
        REPO_ROOT=str(root),
        ANALYTICS_ROOT=str(analytics),
        RAW_DIR=str(analytics / "01_raw"),
        CURATED_DIR=str(analytics / "03_curated"),
    )
