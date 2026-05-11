"""
stage1/refine_clip.py — Stage 1 time flow 후보 클립 정밀화 (clip variant).
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    SKIP_EXISTING, STAGE1_MIN_SEC,
    STAGE2_CLIP_SEC, STAGE2_CLIP_WINDOW_SEC, STAGE2_CLIP_TARGET_FPS, STAGE2_CLIP_MAX_FRAMES,
    STAGE2_MAX_RETRIES,
)
from ..utils.inference import call_qwen_for_frame_sequence
from ..utils.io_utils import (
    append_stage_row, get_artifacts_dir, get_processed_paths,
    load_stage_csv, resolve_video_path,
)
from ..utils.logging_utils import format_elapsed
from ..utils.validators import is_near_zero, snap_time_to_frame, validate_accident_time
from ..utils.video import (
    extract_clip, extract_frame_at_time, load_clip_frames_for_vlm,
    probe_video_info, save_frame_jpg,
)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

def build_stage2_clip_prompt(
    timestamps: List[float],
    candidate: float,
    clip_start: float,
    clip_end: float,
    candidate_rank: int = 1,
    retry_count: int = 0,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    retry_block = ""
    if retry_count > 0:
        retry_block = (
            "\n[CRITICAL NOTE]\n"
            "Your previous answer was near 0 seconds or outside the clip range. "
            f"The clip spans {clip_start:.3f}s to {clip_end:.3f}s. "
            "Your answer must be within this range.\n"
        )

    catalog = ", ".join(f"{i}: {ts:.3f}s" for i, ts in enumerate(timestamps))

    video_section = ""
    if video_info:
        lines = []
        w, h = video_info.get("width", 0), video_info.get("height", 0)
        fps  = video_info.get("fps", 0.0)
        dur  = video_info.get("duration", 0.0)
        if w and h: lines.append(f"- original resolution: {w}x{h}")
        if fps:     lines.append(f"- original fps: {fps:.3f}")
        if dur:     lines.append(f"- full video duration: {dur:.3f}s")
        if lines:
            video_section = "\nVideo properties:\n" + "\n".join(lines) + "\n"

    return f"""{retry_block}You are an expert traffic accident analyst.
You are given a SHORT CLIP extracted from a CCTV video.
{video_section}
Clip range  : {clip_start:.3f}s – {clip_end:.3f}s  (all timestamps below are absolute)
Candidate {candidate_rank}: {candidate:.3f}s  (Stage-1 estimate — may be slightly off)

Frame timestamps: {catalog}

Task: Find the SINGLE MOST PRECISE time when the traffic accident FIRST begins in this clip.

Instructions:
1. All timestamps in this clip are in ABSOLUTE seconds from the start of the full video.
2. The candidate time {candidate:.3f}s is your starting reference — look carefully around it.
3. Identify the EARLIEST frame where physical contact or collision is unambiguously beginning.
4. Output accident_time with THREE decimal places (e.g., {candidate:.3f}).
5. Your answer MUST be between {clip_start:.3f}s and {clip_end:.3f}s.

Critical output rules:
- Output exactly ONE JSON object.
- No markdown, no code fences, no text before or after JSON.
Output format:
{{
  "accident_time": <float, absolute seconds with 3 decimal places>,
  "confidence": <float 0.0–1.0>
}}""".strip()


# ---------------------------------------------------------------------------
# Stage 1 refinement
# ---------------------------------------------------------------------------

def _analyze_clip(
    model, sampling_params,
    abs_path: str, candidate: float, candidate_rank: int,
    video_info: Dict[str, Any], video_name: str,
) -> Tuple[Optional[float], float]:
    duration   = video_info.get("duration", 0.0)
    clip_start = max(0.0, candidate - STAGE2_CLIP_WINDOW_SEC)
    clip_end   = min(duration, candidate + STAGE2_CLIP_WINDOW_SEC)

    sampled = load_clip_frames_for_vlm(abs_path, clip_start, clip_end,
                                        STAGE2_CLIP_TARGET_FPS, STAGE2_CLIP_MAX_FRAMES)
    if not sampled:
        print(f"    [클립 없음] 후보 {candidate_rank}: {candidate:.3f}s")
        return None, 0.0

    print(f"    후보 {candidate_rank} ({candidate:.3f}s): "
          f"클립 [{clip_start:.2f}~{clip_end:.2f}s], {sampled['num_frames']}프레임")

    for attempt in range(STAGE2_MAX_RETRIES + 1):
        prompt = build_stage2_clip_prompt(
            sampled["timestamps"], candidate=candidate,
            clip_start=clip_start, clip_end=clip_end,
            candidate_rank=candidate_rank, retry_count=attempt, video_info=video_info,
        )
        result = call_qwen_for_frame_sequence(
            model, sampling_params, sampled["frames"], sampled["timestamps"],
            prompt, label=f"stage1/refine_clip{candidate_rank} {video_name}", max_retries=2,
        )
        if not result:
            continue

        t = validate_accident_time(result)
        if t is None or is_near_zero(t, STAGE1_MIN_SEC):
            print(f"      [재시도] 0초 근처, 시도 {attempt + 1}/{STAGE2_MAX_RETRIES}")
            continue
        if not (clip_start - 0.5 <= t <= clip_end + 0.5):
            print(f"      [재시도] 범위 벗어남 ({t:.3f}s)")
            continue

        return t, result.get("confidence", 0.5)

    return None, 0.0


def run_stage2(
    model, sampling_params, run_dir: str, video_lookup: Dict[str, str],
) -> None:
    stage1_data = load_stage_csv(run_dir, 1)
    if not stage1_data:
        print("[Stage 1 refinement] stage1.csv 없음 — Stage 1을 먼저 실행하세요.")
        return

    print(f"\n[Stage 1 refinement / Clip] 집중 클리핑 정밀화 — {len(stage1_data)}개 영상")
    processed = get_processed_paths(run_dir, 2) if SKIP_EXISTING else set()

    for video_name, s1_row in stage1_data.items():
        if SKIP_EXISTING and video_name in processed:
            print(f"  [SKIP] {video_name}")
            continue

        abs_path = resolve_video_path(video_name, video_lookup)
        if not abs_path:
            continue

        t0         = time.time()
        video_info = probe_video_info(abs_path)

        def _safe_float(v):
            try:
                f = float(v)
                return f if f > STAGE1_MIN_SEC else None
            except (TypeError, ValueError):
                return None

        c1 = _safe_float(s1_row.get("candidate_1"))
        c2 = _safe_float(s1_row.get("candidate_2"))
        c3 = _safe_float(s1_row.get("candidate_3"))
        best_stage1 = _safe_float(s1_row.get("accident_time"))

        print(f"\n  처리: {video_name}  (Top-3: {[c1, c2, c3]})")

        results = []
        for rank, candidate in enumerate([c1, c2, c3], 1):
            if candidate is None:
                continue
            t_clip, conf = _analyze_clip(model, sampling_params, abs_path, candidate,
                                          rank, video_info, video_name)
            if t_clip is not None:
                results.append((t_clip, conf))
                print(f"    -> 후보 {rank} 결과: {t_clip:.3f}s (conf={conf:.2f})")

        fallback_used = False
        if results:
            results.sort(key=lambda x: -x[1])
            accident_time = results[0][0]
        else:
            accident_time = best_stage1 if best_stage1 else round(
                video_info.get("duration", 0) / 2, 3)
            fallback_used = True
            print(f"  [fallback] {accident_time:.3f}s 사용")

        accident_time = snap_time_to_frame(
            accident_time, video_info["fps"], video_info["total_frames"]
        ) if video_info["fps"] > 0 else accident_time

        frame_index = round(accident_time * video_info["fps"]) if video_info["fps"] > 0 else 0
        clip_start  = max(0.0, accident_time - STAGE2_CLIP_SEC / 2)
        clip_end    = min(video_info.get("duration", accident_time + STAGE2_CLIP_SEC / 2),
                          accident_time + STAGE2_CLIP_SEC / 2)

        stem           = os.path.splitext(video_name)[0]
        artifact_dir   = get_artifacts_dir(run_dir, stem)
        key_frame_path = os.path.join(artifact_dir, "key_frame.jpg")
        clip_path      = os.path.join(artifact_dir, "fixed_clip.mp4")

        frame_result = extract_frame_at_time(abs_path, accident_time)
        if frame_result:
            save_frame_jpg(frame_result["frame"], key_frame_path)
        else:
            key_frame_path = ""

        clip_ok = extract_clip(abs_path, clip_path, clip_start, clip_end)

        print(f"  -> Stage1 refine: {accident_time:.3f}s (fallback={fallback_used})")

        append_stage_row(run_dir, 2, {
            "path"          : video_name,
            "accident_time" : round(accident_time, 4),
            "frame_index"   : frame_index,
            "clip_start"    : round(clip_start, 4),
            "clip_end"      : round(clip_end, 4),
            "key_frame_path": key_frame_path,
            "clip_path"     : clip_path if clip_ok else "",
            "fallback_used" : int(fallback_used),
        })
        print(f"  완료 ({format_elapsed(time.time() - t0)})")
