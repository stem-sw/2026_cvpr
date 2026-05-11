"""
stage1/refine_full.py — Stage 1 time flow 정밀화 (full_run variant).
"""
import os
import time
from typing import Any, Dict, List, Optional

from ..config import (
    DEBUG_VISIBLE_EVIDENCE,
    SKIP_EXISTING, STAGE1_MIN_SEC,
    STAGE2_CLIP_SEC, STAGE2_MAX_FRAMES, STAGE2_MAX_RETRIES, STAGE2_TARGET_FPS,
)
from ..utils.inference import call_qwen_for_frame_sequence
from ..utils.io_utils import (
    append_stage_row, get_artifacts_dir, get_processed_paths,
    load_stage_csv, resolve_video_path,
)
from ..utils.logging_utils import format_elapsed
from ..utils.validators import is_near_zero, print_debug_payload, snap_time_to_frame, validate_accident_time
from ..utils.video import (
    extract_clip, extract_frame_at_time, load_video_frames_for_vlm,
    probe_video_info, save_frame_jpg,
)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

def _video_props(info: Optional[Dict[str, Any]]) -> str:
    if not info:
        return ""
    lines = []
    w, h = info.get("width", 0), info.get("height", 0)
    fps  = info.get("fps", 0.0)
    dur  = info.get("duration", 0.0)
    if w and h: lines.append(f"- resolution: {w}x{h}")
    if fps:     lines.append(f"- fps: {fps:.3f}")
    if dur:     lines.append(f"- duration: {dur:.3f}s")
    return ("\nVideo properties:\n" + "\n".join(lines) + "\n") if lines else ""


def _time_output_format() -> str:
    if DEBUG_VISIBLE_EVIDENCE:
        return ('Output format:\n{\n  "accident_time": <float, seconds with 3 decimal places>,\n'
                '  "confidence": <float 0.0-1.0>,\n  "evidence": "<brief visible cues>"\n}')
    return 'Output format:\n{\n  "accident_time": <float, seconds with 3 decimal places>\n}'


def build_stage2_prompt(
    timestamps: List[float],
    candidate_time: Optional[float],
    retry_count: int = 0,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    retry_block = ""
    if retry_count > 0:
        retry_block = (
            "\n[CRITICAL NOTE]\n"
            "Your previous answer was near 0 seconds, which is almost certainly wrong. "
            "Re-check the entire video from start to end. "
            "Do not collapse to coarse round numbers. Be frame-accurate.\n"
        )
    hint_block = ""
    if candidate_time is not None:
        hint_block = (
            f"\nWeak reference candidate time (may be wrong): {candidate_time:.3f}s\n"
            "- Do NOT rely on this. Use it only as a weak hint.\n"
            "- If this candidate seems wrong, ignore it and judge independently.\n"
        )
    catalog = ", ".join(f"{i}: {ts:.3f}s" for i, ts in enumerate(timestamps))

    return f"""{retry_block}You are an expert traffic accident analyst.
You are given the FULL chronological sequence of CCTV frames.
{_video_props(video_info)}{hint_block}
Frame timestamps: {catalog}

Task: Find the PRECISE time when the traffic accident first begins.

Instructions:
1. Analyze the FULL video carefully from start to end.
2. Identify the earliest frame where physical contact begins, or where impact is
   clearly unavoidable and immediate.
3. Output accident_time with THREE decimal places (e.g., 4.333).
4. The returned time must correspond to the earliest onset frame, not a later peak.
5. Do not output 0 or near-zero time unless the accident genuinely starts at the very beginning.

Critical output rules:
- Output exactly ONE JSON object.
- No markdown, no code fences, no text before or after JSON.
{_time_output_format()}""".strip()


# ---------------------------------------------------------------------------
# Stage 1 refinement
# ---------------------------------------------------------------------------

def run_stage2(
    model, sampling_params, run_dir: str, video_lookup: Dict[str, str],
) -> None:
    stage1_data = load_stage_csv(run_dir, 1)
    if not stage1_data:
        print("[Stage 1 refinement] stage1.csv 없음 — Stage 1을 먼저 실행하세요.")
        return

    print(f"\n[Stage 1 refinement] 사고 시각 정밀화 — {len(stage1_data)}개 영상")
    processed = get_processed_paths(run_dir, 2) if SKIP_EXISTING else set()

    for video_name, s1_row in stage1_data.items():
        if SKIP_EXISTING and video_name in processed:
            print(f"  [SKIP] {video_name}")
            continue

        abs_path = resolve_video_path(video_name, video_lookup)
        if not abs_path:
            print(f"  [오류] 파일 없음: {video_name}")
            continue

        t0         = time.time()
        video_info = probe_video_info(abs_path)
        candidate  = float(s1_row.get("accident_time") or 0)
        print(f"\n  처리: {video_name}  (Stage1 후보: {candidate:.3f}s)")

        sampled = load_video_frames_for_vlm(abs_path, STAGE2_TARGET_FPS, STAGE2_MAX_FRAMES)
        if not sampled:
            _save_fallback(run_dir, video_name, abs_path, candidate, video_info)
            continue

        print(f"  샘플링: {sampled['num_frames']}프레임 (fps={STAGE2_TARGET_FPS})")

        accident_time = None
        fallback_used = False

        for attempt in range(STAGE2_MAX_RETRIES + 1):
            prompt = build_stage2_prompt(sampled["timestamps"], candidate_time=candidate,
                                         retry_count=attempt, video_info=video_info)
            result = call_qwen_for_frame_sequence(
                model, sampling_params, sampled["frames"], sampled["timestamps"],
                prompt, label=f"stage1/refine {video_name}", max_retries=2,
            )
            t = validate_accident_time(result or {})
            print_debug_payload("stage1/refine", result)

            if t is not None and not is_near_zero(t, STAGE1_MIN_SEC):
                accident_time = snap_time_to_frame(
                    t, video_info["fps"], video_info["total_frames"]
                ) if video_info["fps"] > 0 else t
                break
            print(f"  [재시도] 0초 근처, 재시도 {attempt + 1}/{STAGE2_MAX_RETRIES}")

        if accident_time is None:
            accident_time = candidate if candidate > STAGE1_MIN_SEC else round(
                video_info.get("duration", 0) / 2, 3)
            fallback_used = True
            print(f"  [fallback] {accident_time:.3f}s 사용")

        _save_stage2_row(run_dir, video_name, abs_path, accident_time, fallback_used, video_info)
        print(f"  완료 ({format_elapsed(time.time() - t0)})")


def _save_stage2_row(run_dir, video_name, abs_path, accident_time, fallback_used, video_info):
    frame_index = round(accident_time * video_info["fps"]) if video_info["fps"] > 0 else 0
    clip_start  = max(0.0, accident_time - STAGE2_CLIP_SEC / 2)
    clip_end    = min(video_info.get("duration", accident_time + STAGE2_CLIP_SEC / 2),
                      accident_time + STAGE2_CLIP_SEC / 2)

    stem           = os.path.splitext(video_name)[0]
    artifact_dir   = get_artifacts_dir(run_dir, stem)
    key_frame_path = os.path.join(artifact_dir, "key_frame.jpg")
    clip_path      = os.path.join(artifact_dir, "fixed_clip.mp4")

    frame_result = extract_frame_at_time(abs_path, accident_time)
    if frame_result is not None:
        save_frame_jpg(frame_result["frame"], key_frame_path)
    else:
        key_frame_path = ""

    clip_ok = extract_clip(abs_path, clip_path, clip_start, clip_end)
    if not clip_ok:
        clip_path = ""

    print(f"  -> Stage1 refine: {accident_time:.3f}s (frame={frame_index}, fallback={fallback_used})")

    append_stage_row(run_dir, 2, {
        "path"          : video_name,
        "accident_time" : round(accident_time, 4),
        "frame_index"   : frame_index,
        "clip_start"    : round(clip_start, 4),
        "clip_end"      : round(clip_end, 4),
        "key_frame_path": key_frame_path,
        "clip_path"     : clip_path,
        "fallback_used" : int(fallback_used),
    })


def _save_fallback(run_dir, video_name, abs_path, candidate, video_info):
    accident_time = candidate if candidate > STAGE1_MIN_SEC else round(
        video_info.get("duration", 0) / 2, 3)
    _save_stage2_row(run_dir, video_name, abs_path, accident_time, True, video_info)
