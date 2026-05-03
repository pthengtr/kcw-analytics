import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import find_dotenv, load_dotenv
from supabase import create_client


# ==========================================================
# ENV
# ==========================================================

env_path = find_dotenv()
load_dotenv(env_path, override=True)


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(f"Missing required env: {name}")

    return value or ""


SUPABASE_URL = env("SUPABASE_URL", required=True)
SUPABASE_SERVICE_ROLE_KEY = env("SUPABASE_SERVICE_ROLE_KEY", required=True)
SUPABASE_DB_URL = env("SUPABASE_DB_URL", required=True)

BUCKET = env("PRODUCT_IMAGE_BUCKET", "pictures")
BASE_FOLDER = env("PRODUCT_IMAGE_BASE_FOLDER", "product").strip("/")

LEGACY_IMAGE_DIR = Path(env("LEGACY_PRODUCT_IMAGE_DIR", required=True))
DELETE_MODE = env("PRODUCT_IMAGE_DELETE_MODE", "quarantine").strip().lower()

PRODUCT_IMAGE_INDEX_SOURCE = env("PRODUCT_IMAGE_INDEX_SOURCE", "db").strip().lower()

MANIFEST_PATH = LEGACY_IMAGE_DIR / "_product_image_sync_manifest.json"
QUARANTINE_DIR = LEGACY_IMAGE_DIR / "_deleted"

LEGACY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ==========================================================
# BASIC HELPERS
# ==========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_product_filenames(bcode: str) -> list[str]:
    """
    Legacy-aligned filename convention.

    Slot 1 = {bcode}.jpg
    Slot 2 = {bcode}_2.jpg
    Slot 3 = {bcode}_3.jpg
    Slot 4 = {bcode}_4.jpg
    Slot 5 = {bcode}_5.jpg
    """
    return [
        f"{bcode}.jpg",
        f"{bcode}_2.jpg",
        f"{bcode}_3.jpg",
        f"{bcode}_4.jpg",
        f"{bcode}_5.jpg",
    ]


def is_expected_product_image_path(remote_path: str) -> tuple[bool, str | None, str | None]:
    """
    Accept only:
    product/{bcode}/{bcode}.jpg
    product/{bcode}/{bcode}_2.jpg
    product/{bcode}/{bcode}_3.jpg
    product/{bcode}/{bcode}_4.jpg
    product/{bcode}/{bcode}_5.jpg
    """
    parts = remote_path.split("/")

    if len(parts) != 3:
        return False, None, None

    base_folder, bcode, filename = parts

    if base_folder != BASE_FOLDER:
        return False, None, None

    if filename not in expected_product_filenames(bcode):
        return False, None, None

    return True, bcode, filename


def local_path_from_filename(filename: str) -> Path:
    return LEGACY_IMAGE_DIR / filename


def safe_iso(value: Any) -> str:
    if value is None:
        return ""

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def metadata_size(metadata: Any) -> int | None:
    if not isinstance(metadata, dict):
        return None

    size = metadata.get("size")

    try:
        return int(size) if size is not None else None
    except Exception:
        return None


def atomic_write_bytes(target_path: Path, data: bytes):
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        f.write(data)

    os.replace(tmp_path, target_path)


# ==========================================================
# MANIFEST
# ==========================================================

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "version": 1,
            "last_run_at": None,
            "files": {},
        }

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict):
    manifest["last_run_at"] = now_iso()

    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, MANIFEST_PATH)


# ==========================================================
# FAST REMOTE INDEX FROM storage.objects
# ==========================================================

def build_remote_index_from_db() -> dict[str, dict]:
    """
    Fast path:
    Read Storage object metadata directly from Postgres instead of calling
    Supabase Storage list() for every product folder.
    """
    result: dict[str, dict] = {}

    sql = """
        select
            name,
            created_at,
            updated_at,
            metadata
        from storage.objects
        where bucket_id = %s
          and name like %s
        order by name
    """

    print("Building remote image index from storage.objects...")
    print("Bucket:", BUCKET)
    print("Prefix:", f"{BASE_FOLDER}/%")

    with psycopg.connect(SUPABASE_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (BUCKET, f"{BASE_FOLDER}/%"))

            row_count = 0
            accepted_count = 0

            for name, created_at, updated_at, metadata in cur:
                row_count += 1

                remote_path = str(name or "").strip()
                if not remote_path:
                    continue

                ok, bcode, filename = is_expected_product_image_path(remote_path)
                if not ok or not bcode or not filename:
                    continue

                result[remote_path] = {
                    "remote_path": remote_path,
                    "bcode": bcode,
                    "supabase_filename": filename,
                    "local_filename": filename,
                    "updated_at": safe_iso(updated_at),
                    "created_at": safe_iso(created_at),
                    "size": metadata_size(metadata),
                }

                accepted_count += 1

            print(f"storage.objects rows scanned: {row_count:,}")
            print(f"valid product images accepted: {accepted_count:,}")

    return result


# ==========================================================
# OPTIONAL SLOW FALLBACK
# ==========================================================

def build_remote_index() -> dict[str, dict]:
    if PRODUCT_IMAGE_INDEX_SOURCE == "db":
        return build_remote_index_from_db()

    raise ValueError(
        f"Unsupported PRODUCT_IMAGE_INDEX_SOURCE={PRODUCT_IMAGE_INDEX_SOURCE}. "
        "Use PRODUCT_IMAGE_INDEX_SOURCE=db."
    )


# ==========================================================
# SYNC DECISION
# ==========================================================

def needs_download(remote_path: str, remote_info: dict, manifest: dict) -> bool:
    local_filename = remote_info["local_filename"]
    local_path = local_path_from_filename(local_filename)

    if not local_path.exists():
        return True

    known = manifest.get("files", {}).get(remote_path)

    if not known:
        return True

    if known.get("updated_at") != remote_info.get("updated_at"):
        return True

    if known.get("size") != remote_info.get("size"):
        return True

    return False


def quarantine_local_file(local_path: Path):
    if not local_path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = QUARANTINE_DIR / f"{local_path.stem}_{timestamp}{local_path.suffix}"

    shutil.move(str(local_path), str(target))


def download_remote_file(remote_path: str) -> bytes:
    return supabase.storage.from_(BUCKET).download(remote_path)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("Loaded .env:", env_path)
    print("Legacy image dir:", LEGACY_IMAGE_DIR)
    print("Delete mode:", DELETE_MODE)

    manifest = load_manifest()
    files_manifest = manifest.setdefault("files", {})

    remote_index = build_remote_index()

    to_download = [
        (remote_path, info)
        for remote_path, info in remote_index.items()
        if needs_download(remote_path, info, manifest)
    ]

    previous_remote_paths = set(files_manifest.keys())
    current_remote_paths = set(remote_index.keys())
    to_remove = sorted(previous_remote_paths - current_remote_paths)

    report = {
        "started_at": now_iso(),
        "remote_files_found": len(remote_index),
        "to_download": len(to_download),
        "to_remove": len(to_remove),
        "downloaded": 0,
        "removed": 0,
        "quarantined": 0,
        "skipped": len(remote_index) - len(to_download),
        "failed": [],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    # --------------------------
    # Download missing/changed
    # --------------------------
    for idx, (remote_path, info) in enumerate(to_download, start=1):
        local_path = local_path_from_filename(info["local_filename"])

        try:
            print(
                f"[{idx}/{len(to_download)}] "
                f"{remote_path} -> {local_path.name}"
            )

            data = download_remote_file(remote_path)
            atomic_write_bytes(local_path, data)

            files_manifest[remote_path] = {
                "remote_path": remote_path,
                "bcode": info["bcode"],
                "supabase_filename": info["supabase_filename"],
                "local_filename": info["local_filename"],
                "local_path": str(local_path),
                "updated_at": info["updated_at"],
                "created_at": info["created_at"],
                "size": info["size"],
                "last_synced_at": now_iso(),
            }

            report["downloaded"] += 1

        except Exception as e:
            print(f"[ERROR] Download failed for {remote_path}: {e}")
            report["failed"].append(
                {
                    "remote_path": remote_path,
                    "error": str(e),
                }
            )

    # --------------------------
    # Remove/quarantine deleted
    # --------------------------
    for remote_path in to_remove:
        known = files_manifest.get(remote_path, {})
        local_filename = known.get("local_filename")

        if not local_filename:
            files_manifest.pop(remote_path, None)
            continue

        local_path = local_path_from_filename(local_filename)

        try:
            print(f"[REMOVE] {remote_path} -> {local_filename}")

            if DELETE_MODE == "quarantine":
                quarantine_local_file(local_path)
                report["quarantined"] += 1

            elif DELETE_MODE == "delete":
                if local_path.exists():
                    local_path.unlink()
                report["removed"] += 1

            elif DELETE_MODE == "ignore":
                pass

            else:
                raise ValueError(f"Unsupported PRODUCT_IMAGE_DELETE_MODE: {DELETE_MODE}")

            files_manifest.pop(remote_path, None)

        except Exception as e:
            print(f"[ERROR] Remove/quarantine failed for {remote_path}: {e}")
            report["failed"].append(
                {
                    "remote_path": remote_path,
                    "local_filename": local_filename,
                    "error": str(e),
                }
            )

    save_manifest(manifest)

    report["finished_at"] = now_iso()
    report["manifest_files"] = len(files_manifest)

    report_path = LEGACY_IMAGE_DIR / f"_product_image_sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Report:", report_path)


if __name__ == "__main__":
    main()