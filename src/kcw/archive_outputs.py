"""Archive 04_outputs to 06_archive as export_*.tar.gz.

The folder is ~5k files / ~160MB. Re-downloading it from Drive on every Linux
run is what made 00 crawl. Windows HQ-PC is fast because Drive File Stream
already has the tree on disk.

Linux: rclone sync into a persistent local cache, tar.gz that cache, upload
one archive file. First sync is still a one-time fill; later days are deltas.
Windows: tarfile from the Drive path, copy one .tar.gz into 06_archive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from src.kcw.paths import analytics_root

EXCLUDE_NAMES = frozenset({"desktop.ini", "Thumbs.db", ".DS_Store"})
DEFAULT_RCLONE_SPEC = "kcw,team_drive=0AJ5BTDhgit7-Uk9PVA"


def _is_rclone_fuse(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            mounts = fh.readlines()
    except OSError:
        return False
    best = ""
    fuse = False
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt, fstype = parts[1], parts[2]
        if resolved == Path(mnt) or str(resolved).startswith(mnt.rstrip("/") + "/"):
            if len(mnt) >= len(best):
                best = mnt
                fuse = "rclone" in fstype or fstype == "fuse.rclone"
    return fuse


def _rclone_bin() -> str:
    return os.environ.get("KCW_RCLONE", shutil.which("rclone") or "rclone")


def _rclone_spec() -> str:
    return os.environ.get("KCW_RCLONE_SPEC", DEFAULT_RCLONE_SPEC)


def _remote_rel(inner: str) -> str:
    return "kcw_analytics/" + inner


def _cache_dir() -> Path:
    env = os.environ.get("KCW_ARCHIVE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "kcw" / "04_outputs"


def _rclone(args: list[str]) -> None:
    cmd = [_rclone_bin(), *args]
    print("[archive]", " ".join(cmd[:8]), "...")
    subprocess.run(cmd, check=True)


def write_tar_gz(src_dir: Path, dest_tar: Path) -> None:
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    root = src_dir.resolve()
    with tarfile.open(dest_tar, "w:gz") as tar:
        for path in root.rglob("*"):
            if not path.is_file() or path.name in EXCLUDE_NAMES:
                continue
            tar.add(path, arcname=str(Path("04_outputs") / path.relative_to(root)))


def _sync_outputs_to_cache(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    spec = _rclone_spec()
    _rclone(
        [
            "sync",
            "%s:%s" % (spec, _remote_rel("04_outputs")),
            str(cache),
            "--fast-list",
            "--exclude",
            "desktop.ini",
            "--exclude",
            "Thumbs.db",
            "--exclude",
            ".DS_Store",
            "--transfers",
            "16",
            "--checkers",
            "32",
        ]
    )


def archive_04_outputs(*, analytics: Path | None = None) -> Path:
    """Write 06_archive/export_YYYY-MM-DD-HHMM.tar.gz."""
    root = analytics or analytics_root()
    src = root / "04_outputs"
    dest_dir = root / "06_archive"
    suffix = datetime.now().strftime("%Y-%m-%d-%H%M")
    name = "export_%s.tar.gz" % suffix
    dest = dest_dir / name

    work = Path(tempfile.mkdtemp(prefix="kcw_archive_"))
    local_tar = work / name
    try:
        if _is_rclone_fuse(src):
            cache = _cache_dir()
            print("[archive] sync 04_outputs ->", cache)
            _sync_outputs_to_cache(cache)
            print("[archive] tar.gz locally", local_tar)
            write_tar_gz(cache, local_tar)
            spec = _rclone_spec()
            _rclone(
                [
                    "copyto",
                    str(local_tar),
                    "%s:%s/%s" % (spec, _remote_rel("06_archive"), name),
                ]
            )
            print("[archive] Created:", dest)
            return dest

        print("[archive] tar.gz from", src)
        write_tar_gz(src, local_tar)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_tar, dest)
        print("[archive] Created:", dest)
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)
