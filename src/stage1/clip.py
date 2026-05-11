"""
stage1/clip.py — Top-3 후보 시각 추출 (clip variant).
"""
import time
from typing import Any, Dict, List, Optional

from ..config import (
    DEBUG_VISIBLE_EVIDENCE,
    SKIP_EXISTING, STAGE1_MAX_FRAMES, STAGE1_MAX_RETRIES,
    STAGE1_MIN_SEC, STAGE1_TARGET_FPS,
)
from ..utils.inference import call_qwen_for_frame_sequence
from ..utils.io_utils import append_stage_row, get_processed_paths, resolve_video_path
from ..utils.logging_utils import format_elapsed
from ..utils.validators import snap_time_to_frame
from ..utils.video import load_video_frames_for_vlm, probe_video_info
from .refine_clip import run_stage2


# ---------------------------------------------------------------------------
# 프롬프트 (Top-3 형식)
# ---------------------------------------------------------------------------

def build_stage1_prompt(
    timestamps: List[float],
    retry_count: int = 0,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    retry_block = ""
    if retry_count > 0:
        retry_block = (
            "\n[CRITICAL NOTE]\n"
            "Your previous answer contained times near 0 seconds. "
            "0 seconds means the very first frame — accidents almost never start at t=0. "
            "Re-examine the frames carefully. All candidates must be > 0.1 seconds.\n"
        )

    catalog = ", ".join(f"{i}: {ts:.3f}s" for i, ts in enumerate(timestamps))

    video_section = ""
    if video_info:
        lines = []
        w, h = video_info.get("width", 0), video_info.get("height", 0)
        fps  = video_info.get("fps", 0.0)
        dur  = video_info.get("duration", 0.0)
        if w and h: lines.append(f"- resolution: {w}x{h}")
        if fps:     lines.append(f"- fps: {fps:.3f}")
        if dur:     lines.append(f"- duration: {dur:.3f}s")
        if lines:
            video_section = "\nVideo properties:\n" + "\n".join(lines) + "\n"

    if DEBUG_VISIBLE_EVIDENCE:
        output_format = ('Output format:\n{\n  "candidates": [\n'
                         '    {"time": <float>, "confidence": <float 0-1>, "evidence": "<cue>"},\n'
                         '    {"time": <float>, "confidence": <float 0-1>, "evidence": "<cue>"},\n'
                         '    {"time": <float>, "confidence": <float 0-1>, "evidence": "<cue>"}\n  ]\n}')
    else:
        output_format = ('Output format:\n{\n  "candidates": [\n'
                         '    {"time": <float, seconds>, "confidence": <float 0-1>},\n'
                         '    {"time": <float, seconds>, "confidence": <float 0-1>},\n'
                         '    {"time": <float, seconds>, "confidence": <float 0-1>}\n  ]\n}')

    return f"""{retry_block}You are an expert traffic accident analyst.
You are given a chronological sequence of CCTV frames sampled from a full video.
{video_section}
Frame timestamps: {catalog}

Task: Find exactly 3 candidate times for when a traffic accident could begin in this video.

Instructions:
1. Analyze ALL frames carefully from start to end.
2. Find 3 candidate times that are each at least 1.0 second apart from each other.
3. Rank them by confidence (highest first).
4. Candidate 1 should be your best guess for the actual accident onset.
5. Candidates 2 and 3 should be OTHER plausible moments.
6. Do NOT repeat the same time. Each candidate must differ by at least 1.0 second.
7. All times must be > 0.1 seconds and within the video duration.

Critical output rules:
- Output exactly ONE JSON object with a "candidates" array of exactly 3 items.
- No markdown, no code fences, no text before or after JSON.
{output_format}""".strip()


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def _parse_top3_candidates(result: Optional[Dict[str, Any]]) -> List[Optional[float]]:
    if not result:
        return [None, None, None]
    candidates = result.get("candidates", [])
    times = []
    for c in candidates:
        try:
            t = float(c.get("time") or c.get("accident_time") or 0)
            if t <= STAGE1_MIN_SEC:
                continue
            if any(existing is not None and abs(t - existing) < 1.0 for existing in times):
                continue
            times.append(t)
        except (TypeError, ValueError):
            continue
    while len(times) < 3:
        times.append(None)
    return times[:3]


def run_stage1(
    model, sampling_params, run_dir: str,
    video_files: List[str], video_lookup: Dict[str, str],
    max_frames: Optional[int] = None,
) -> None:
    effective_max_frames = max_frames if max_frames is not None else STAGE1_MAX_FRAMES
    print(f"\n[Stage 1 / Top3] 사고 시각 Top-3 추출 — {len(video_files)}개 영상 "
          f"(max_frames={effective_max_frames})")
    processed = get_processed_paths(run_dir, 1) if SKIP_EXISTING else set()

    for video_name in video_files:
        if SKIP_EXISTING and video_name in processed:
            print(f"  [SKIP] {video_name}")
            continue

        abs_path = resolve_video_path(video_name, video_lookup)
        if not abs_path:
            print(f"  [오류] 파일 없음: {video_name}")
            continue

        t0         = time.time()
        video_info = probe_video_info(abs_path)
        print(f"\n  처리: {video_name}  ({video_info.get('duration', 0):.1f}s)")

        sampled = load_video_frames_for_vlm(abs_path, STAGE1_TARGET_FPS, effective_max_frames)
        if not sampled:
            append_stage_row(run_dir, 1, {
                "path": video_name, "accident_time": "", "frame_index": "",
                "candidate_1": "", "candidate_2": "", "candidate_3": "", "retry_count": 0,
            })
            continue

        print(f"  샘플링: {sampled['num_frames']}프레임 (fps={STAGE1_TARGET_FPS})")

        candidates  = [None, None, None]
        retry_count = 0

        for attempt in range(STAGE1_MAX_RETRIES + 1):
            prompt = build_stage1_prompt(sampled["timestamps"], retry_count=attempt,
                                         video_info=video_info)
            result = call_qwen_for_frame_sequence(
                model, sampling_params, sampled["frames"], sampled["timestamps"],
                prompt, label=f"stage1/top3 {video_name}", max_retries=2,
            )
            parsed = _parse_top3_candidates(result)
            valid  = [t for t in parsed if t is not None and t > STAGE1_MIN_SEC]

            if valid:
                candidates  = parsed
                retry_count = attempt
                break
            print(f"  [재시도] 유효 후보 없음, 재시도 {attempt + 1}/{STAGE1_MAX_RETRIES}")

        best = next((t for t in candidates if t is not None and t > STAGE1_MIN_SEC), None)
        if best is None:
            best = round(video_info.get("duration", 0) / 2, 3)
            print(f"  [fallback] {best:.3f}s 사용")

        best = snap_time_to_frame(best, video_info["fps"], video_info["total_frames"]) \
               if video_info["fps"] > 0 else best
        frame_index = round(best * video_info["fps"]) if video_info["fps"] > 0 else 0

        print(f"  -> Top-3: {[round(t, 3) if t else None for t in candidates]}")
        print(f"  -> best={best:.3f}s (frame={frame_index})")

        append_stage_row(run_dir, 1, {
            "path"         : video_name,
            "accident_time": round(best, 4),
            "frame_index"  : frame_index,
            "candidate_1"  : round(candidates[0], 4) if candidates[0] else "",
            "candidate_2"  : round(candidates[1], 4) if candidates[1] else "",
            "candidate_3"  : round(candidates[2], 4) if candidates[2] else "",
            "retry_count"  : retry_count,
        })
        print(f"  완료 ({format_elapsed(time.time() - t0)})")


def run_stage1_flow(
    model, stage1_sampling_params, sampling_params, run_dir: str,
    video_files: List[str], video_lookup: Dict[str, str],
    max_frames: Optional[int] = None,
) -> None:
    """Top-3 후보 추출부터 집중 클립 정밀화(stage2.csv 생성)까지 한 번에 수행합니다."""
    run_stage1(
        model, stage1_sampling_params, run_dir,
        video_files, video_lookup, max_frames=max_frames,
    )
    run_stage2(model, sampling_params, run_dir, video_lookup)
