#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
REPO_ROOT="$PROJECT_ROOT"
DATA_ROOT="${ACCIDENT_DATA_ROOT:-$PROJECT_ROOT/data}"
DOWNLOAD_ROOT="${ACCIDENT_DOWNLOAD_ROOT:-$DATA_ROOT/downloads}"
RAW_DATA_ROOT="${ACCIDENT_RAW_DATA_ROOT:-$DATA_ROOT/raw}"
VIDEO_DIR="${ACCIDENT_VIDEO_DIR:-$DATA_ROOT/videos}"
MANIFEST_PATH="${ACCIDENT_VIDEO_MANIFEST:-$DATA_ROOT/video_manifest.csv}"
ARCHIVE_PATH="${1:-${ACCIDENT_ARCHIVE_PATH:-}}"
EXPECTED_VIDEO_COUNT=2027

echo "========================================="
echo " Accident Dataset Setup"
echo "========================================="
echo "[Info] project root : $PROJECT_ROOT"
echo "[Info] repo root    : $REPO_ROOT"
echo "[Info] data root    : $DATA_ROOT"
echo "[Info] zip root     : $DOWNLOAD_ROOT"
echo "[Info] raw root     : $RAW_DATA_ROOT"
echo "[Info] video dir    : $VIDEO_DIR"
echo "[Info] manifest     : $MANIFEST_PATH"
echo "[Info] place accident.zip at $PROJECT_ROOT/accident.zip, or pass a zip path as the first argument."
echo "[Info] make sure you have enough free disk space for the zip, extracted raw files, and indexed videos."

if ! command -v unzip >/dev/null 2>&1; then
    echo "[Error] unzip command could not be found."
    echo "Install unzip and rerun this script."
    exit 1
fi

mkdir -p "$DOWNLOAD_ROOT" "$RAW_DATA_ROOT" "$VIDEO_DIR"
if [[ -z "$ARCHIVE_PATH" ]]; then
    if [[ -f "$REPO_ROOT/accident.zip" ]]; then
        ARCHIVE_PATH="$REPO_ROOT/accident.zip"
    elif [[ -f "$DOWNLOAD_ROOT/accident.zip" ]]; then
        ARCHIVE_PATH="$DOWNLOAD_ROOT/accident.zip"
    else
        first_zip="$(find "$DOWNLOAD_ROOT" -maxdepth 1 -type f -name '*.zip' | sort | head -n 1 || true)"
        ARCHIVE_PATH="${first_zip:-}"
    fi
fi

if [[ -z "$ARCHIVE_PATH" || ! -f "$ARCHIVE_PATH" ]]; then
    echo "[Error] Dataset zip was not found."
    echo "Place accident.zip in: $PROJECT_ROOT/accident.zip"
    echo "Or run: bash setup_data.sh /absolute/path/to/your_dataset.zip"
    exit 1
fi

ZIP_PATH="$(cd "$(dirname "$ARCHIVE_PATH")" && pwd)/$(basename "$ARCHIVE_PATH")"
ARCHIVE_NAME="$(basename "$ZIP_PATH")"
ARCHIVE_STEM="${ARCHIVE_NAME%.zip}"
EXTRACT_ROOT="$RAW_DATA_ROOT/$ARCHIVE_STEM"

mkdir -p "$DOWNLOAD_ROOT" "$EXTRACT_ROOT" "$VIDEO_DIR"
echo "[Info] Using archive: $ZIP_PATH"

echo "[Info] Extracting archive to: $EXTRACT_ROOT"
unzip -qo "$ZIP_PATH" -d "$EXTRACT_ROOT"

echo "[Info] Preparing indexed video directory..."
rm -f "$VIDEO_DIR"/*.mp4
rm -f "$MANIFEST_PATH"

DATA_ROOT="$DATA_ROOT" EXTRACT_ROOT="$EXTRACT_ROOT" VIDEO_DIR="$VIDEO_DIR" MANIFEST_PATH="$MANIFEST_PATH" EXPECTED_VIDEO_COUNT="$EXPECTED_VIDEO_COUNT" python - <<'PY'
import csv
import os
import shutil
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
extract_root = Path(os.environ["EXTRACT_ROOT"])
video_dir = Path(os.environ["VIDEO_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
expected_count = int(os.environ["EXPECTED_VIDEO_COUNT"])
video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

preferred_video_dir = extract_root / "videos"
if preferred_video_dir.is_dir():
    all_videos = sorted(
        [p for p in preferred_video_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts],
        key=lambda p: p.name,
    )
    print(f"[Info] Using preferred video directory: {preferred_video_dir}")
else:
    all_videos = sorted(
        [p for p in extract_root.rglob("*") if p.is_file() and p.suffix.lower() in video_exts],
        key=lambda p: p.name,
    )
    print("[Warn] Top-level videos/ directory was not found. Falling back to recursive scan.")

if not all_videos:
    raise SystemExit("[Error] No video files were found after extraction.")

name_counts = {}
for path in all_videos:
    name_counts[path.name] = name_counts.get(path.name, 0) + 1
duplicates = sorted(name for name, count in name_counts.items() if count > 1)
if duplicates:
    raise SystemExit(
        "[Error] Duplicate original filenames found. Cannot build a deterministic index.\n"
        + "\n".join(duplicates[:20])
    )

video_dir.mkdir(parents=True, exist_ok=True)

rows = []
for index, src in enumerate(all_videos):
    indexed_name = f"{index:04d}_{src.name}"
    dst = video_dir / indexed_name
    link_mode = "hardlink"
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        link_mode = "copy"
    rows.append(
        {
            "index": index,
            "indexed_name": indexed_name,
            "original_name": src.name,
            "source_path": str(src.relative_to(data_root)),
            "materialization": link_mode,
        }
    )

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["index", "indexed_name", "original_name", "source_path", "materialization"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[Info] Indexed {len(rows)} videos into {video_dir}")
print(f"[Info] Wrote manifest to {manifest_path}")
if len(rows) != expected_count:
    print(
        f"[Warn] Expected about {expected_count} videos, but found {len(rows)}. "
        "Double-check the Kaggle download contents."
    )
PY

echo "========================================="
echo " Dataset setup is complete."
echo " Videos are ready in: $VIDEO_DIR"
echo "========================================="
