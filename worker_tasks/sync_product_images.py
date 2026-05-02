import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required env: {name}")
    return value or ""


SUPABASE_URL = env("SUPABASE_URL", required=True)
SUPABASE_SERVICE_ROLE_KEY = env("SUPABASE_SERVICE_ROLE_KEY", required=True)

BUCKET = env("PRODUCT_IMAGE_BUCKET", "pictures")
BASE_FOLDER = env("PRODUCT_IMAGE_BASE_FOLDER", "product").strip("/")

LEGACY_IMAGE_DIR = Path(env("LEGACY_PRODUCT_IMAGE_DIR", required=True))
DELETE_MODE = env("PRODUCT_IMAGE_DELETE_MODE", "quarantine").strip().lower()
LIST_PAGE_SIZE = int(env("PRODUCT_IMAGE_LIST_PAGE_SIZE", "1000"))

MANIFEST_PATH = LEGACY_IMAGE_DIR / "_product_image_sync_manifest.json"
QUARANTINE_DIR = LEGACY_IMAGE_DIR / "_deleted"

LEGACY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_supabase_filenames(bcode: str) -> list[str]:
    return [
        f"{bcode}.jpg",
        f"{bcode}_2.jpg",
        f"{bcode}_3.jpg",
        f"{bcode}_4.jpg",
        f"{bcode}_5.jpg",
    ]


def local_filename_from_remote(bcode: str, filename: str) -> str:
    return filename


def remote_folder_for_bcode(bcode: str) -> str:
    return f"{BASE_FOLDER}/{bcode}"


def remote_path_for_file(bcode: str, filename: str) -> str:
    return f"{remote_folder_for_bcode(bcode)}/{filename}"


def item_timestamp(item: dict) -> str:
    return (
        item.get("updated_at")
        or item.get("created_at")
        or item.get("last_modified")
        or ""
    )

def item_size(item: dict) -> int | None:
    metadata = item.get("metadata") or {}
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


def list_all_items(path: str) -> list[dict]:
    all_items: list[dict] = []
    offset = 0

    while True:
        items = supabase.storage.from_(BUCKET).list(
            path,
            {
                "limit": LIST_PAGE_SIZE,
                "offset": offset,
                "sortBy": {"column": "name", "order": "asc"},
            },
        )

        if not items:
            break

        all_items.extend(items)

        if len(items) < LIST_PAGE_SIZE:
            break

        offset += LIST_PAGE_SIZE

    return all_items


def list_product_bcodes() -> list[str]:
    items = list_all_items(BASE_FOLDER)
    bcodes: list[str] = []

    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue

        # accept folders only; ignore files directly under product/
        if "." in name:
            continue

        bcodes.append(name)

    return sorted(set(bcodes))


def build_remote_index() -> dict[str, dict]:
    result: dict[str, dict] = {}

    bcodes = list_product_bcodes()
    print(f"Found product folders: {len(bcodes):,}")

    for i, bcode in enumerate(bcodes, start=1):
        folder = remote_folder_for_bcode(bcode)
        expected_names = set(expected_supabase_filenames(bcode))

        try:
            items = list_all_items(folder)
        except Exception as e:
            print(f"[WARN] Cannot list folder {folder}: {e}")
            continue

        for item in items:
            name = str(item.get("name") or "").strip()
            if name not in expected_names:
                continue

            remote_path = remote_path_for_file(bcode, name)
            local_filename = local_filename_from_remote(bcode, name)

            result[remote_path] = {
                "remote_path": remote_path,
                "bcode": bcode,
                "supabase_filename": name,
                "local_filename": local_filename,
                "updated_at": item_timestamp(item),
                "size": item_size(item),
            }

        if i % 500 == 0:
            print(f"Scanned {i:,}/{len(bcodes):,} product folders")

    return result


def needs_download(remote_path: str, remote_info: dict, manifest: dict) -> bool:
    local_filename = remote_info["local_filename"]
    local_path = LEGACY_IMAGE_DIR / local_filename

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


def main():
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

    for idx, (remote_path, info) in enumerate(to_download, start=1):
        local_path = LEGACY_IMAGE_DIR / info["local_filename"]

        try:
            print(
                f"[{idx}/{len(to_download)}] "
                f"{remote_path} -> {local_path.name}"
            )

            data = supabase.storage.from_(BUCKET).download(remote_path)
            atomic_write_bytes(local_path, data)

            files_manifest[remote_path] = {
                "remote_path": remote_path,
                "bcode": info["bcode"],
                "supabase_filename": info["supabase_filename"],
                "local_filename": info["local_filename"],
                "local_path": str(local_path),
                "updated_at": info["updated_at"],
                "size": info["size"],
                "last_synced_at": now_iso(),
            }

            report["downloaded"] += 1

        except Exception as e:
            print(f"[ERROR] Download failed for {remote_path}: {e}")
            report["failed"].append(
                {"remote_path": remote_path, "error": str(e)}
            )

    for remote_path in to_remove:
        known = files_manifest.get(remote_path, {})
        local_filename = known.get("local_filename")
        if not local_filename:
            continue

        local_path = LEGACY_IMAGE_DIR / local_filename

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

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()