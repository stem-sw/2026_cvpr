"""
io_utils.py — 실험 디렉터리 관리, 단계별 CSV 읽기/쓰기, 아티팩트 경로 관리.
"""
import csv
import os
import re
from typing import Any, Dict, List, Optional, Set

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

# 단계별 CSV 컬럼 스키마
STAGE_FIELDNAMES = {
    1: ["path", "accident_time", "frame_index", "candidate_1", "candidate_2", "candidate_3", "retry_count"],
    2: ["path", "accident_time", "frame_index", "clip_start", "clip_end",
        "key_frame_path", "clip_path", "fallback_used"],
    3: ["path", "type", "is_single", "evidence"],
    4: ["path", "center_x", "center_y"],
}
SUBMISSION_FIELDNAMES = ["path", "accident_time", "center_x", "center_y", "type"]


# ---------------------------------------------------------------------------
# 경로 헬퍼
# ---------------------------------------------------------------------------

def get_run_dir(output_root: str, run_id: str) -> str:
    return os.path.join(output_root, run_id)


def get_stage_csv_path(run_dir: str, stage: int) -> str:
    return os.path.join(run_dir, f"stage{stage}.csv")


def get_submission_path(run_dir: str) -> str:
    return os.path.join(run_dir, "submission.csv")


def get_artifacts_dir(run_dir: str, video_stem: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', video_stem)
    return os.path.join(run_dir, "artifacts", safe)


# ---------------------------------------------------------------------------
# 비디오 스캔
# ---------------------------------------------------------------------------

def scan_video_files(video_dir: str) -> List[str]:
    """video_dir 내 비디오 파일명 리스트를 정렬하여 반환합니다."""
    entries = []
    for name in os.listdir(video_dir):
        if os.path.splitext(name)[1].lower() not in VIDEO_EXTENSIONS:
            continue
        if os.path.isfile(os.path.join(video_dir, name)):
            entries.append(name)
    entries.sort()
    return entries


def build_video_lookup(video_dir: str) -> Dict[str, str]:
    """파일명 → 절대경로 dict를 생성합니다. 4자리 접두어 제거 버전도 포함."""
    lookup: Dict[str, str] = {}
    for name in os.listdir(video_dir):
        full = os.path.abspath(os.path.join(video_dir, name))
        if not os.path.isfile(full):
            continue
        lookup.setdefault(name, full)
        stripped = re.sub(r"^\d{4}_", "", name) or re.sub(r"^\d+_", "", name)
        if stripped != name:
            lookup.setdefault(stripped, full)
    return lookup


def resolve_video_path(filename: str, lookup: Dict[str, str]) -> Optional[str]:
    return lookup.get(os.path.basename(filename))


# ---------------------------------------------------------------------------
# 단계별 CSV 읽기 / 쓰기
# ---------------------------------------------------------------------------

def load_stage_csv(run_dir: str, stage: int) -> Dict[str, Dict[str, Any]]:
    """stage{N}.csv를 {path: row_dict} 형태로 읽습니다."""
    csv_path = get_stage_csv_path(run_dir, stage)
    if not os.path.exists(csv_path):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("path")
            if p:
                result[p] = dict(row)
    return result


def append_stage_row(run_dir: str, stage: int, row: Dict[str, Any]) -> None:
    """처리 직후 stage{N}.csv에 즉시 1행 append합니다."""
    fieldnames = STAGE_FIELDNAMES[stage]
    csv_path   = get_stage_csv_path(run_dir, stage)
    os.makedirs(run_dir, exist_ok=True)
    needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_submission_row(run_dir: str, row: Dict[str, Any]) -> None:
    """submission.csv에 1행 append합니다."""
    csv_path     = get_submission_path(run_dir)
    needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUBMISSION_FIELDNAMES})


def get_processed_paths(run_dir: str, stage: int) -> Set[str]:
    """stage{N}.csv에서 이미 처리된 path 집합을 반환합니다."""
    data = load_stage_csv(run_dir, stage)
    return set(data.keys())
