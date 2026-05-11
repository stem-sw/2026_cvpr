"""
stage1/full.py — 단일 사고 시각 추론 (full_run variant).
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
from ..utils.validators import is_near_zero, print_debug_payload, snap_time_to_frame, validate_accident_time
from ..utils.video import load_video_frames_for_vlm, probe_video_info


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


def build_stage1_prompt(
    timestamps: List[float],
    retry_count: int = 0,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    retry_block = ""
    if retry_count > 0:
        retry_block = (
            "\n[CRITICAL NOTE]\n"
            "Your previous answer contained an invalid time near 0 seconds. "
            "0 seconds means the very first frame — accidents almost never start at t=0. "
            "Re-examine the frames carefully and output a time > 0.1 seconds.\n"
        )
    catalog = ", ".join(f"{i}: {ts:.3f}s" for i, ts in enumerate(timestamps))
    return f"""{retry_block}You are an expert traffic accident analyst.
You are given a chronological sequence of CCTV frames sampled from a full video.
{_video_props(video_info)}
Frame timestamps: {catalog}

Task: Find the FIRST moment in the video where a traffic accident clearly begins.

Instructions:
1. Analyze the frames in chronological order.
2. Output accident_time as the absolute time (in seconds) when physical contact first begins,
   or when collision is clearly unavoidable and immediate.
3. Use the frame timestamps as reference — your answer must be within the video duration.
4. Do not round to whole seconds unless the accident truly starts at a whole second.
5. Focus only on detecting the accident time. Ignore location and type for now.

Critical output rules:
- Output exactly ONE JSON object.
- No markdown, no code fences, no text before or after JSON.
{_time_output_format()}""".strip()


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(
    model, sampling_params, run_dir: str,
    video_files: List[str], video_lookup: Dict[str, str],
    max_frames: Optional[int] = None,
) -> None:
    effective_max_frames = max_frames if max_frames is not None else STAGE1_MAX_FRAMES
    print(f"\n[Stage 1] 사고 시각 추론 — {len(video_files)}개 영상 (max_frames={effective_max_frames})")
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
        print(f"\n  처리: {video_name}  ({video_info.get('duration', 0):.1f}s, "
              f"{video_info.get('fps', 0):.1f}fps)")

        sampled = load_video_frames_for_vlm(abs_path, STAGE1_TARGET_FPS, effective_max_frames)
        if not sampled:
            print(f"  [오류] 샘플링 실패: {video_name}")
            append_stage_row(run_dir, 1, {"path": video_name, "accident_time": "",
                                          "frame_index": "", "retry_count": 0})
            continue

        print(f"  샘플링: {sampled['num_frames']}프레임 (fps={STAGE1_TARGET_FPS})")

        accident_time = None
        retry_count   = 0

        for attempt in range(STAGE1_MAX_RETRIES + 1):
            prompt = build_stage1_prompt(sampled["timestamps"], retry_count=attempt,
                                         video_info=video_info)
            result = call_qwen_for_frame_sequence(
                model, sampling_params, sampled["frames"], sampled["timestamps"],
                prompt, label=f"stage1 {video_name}", max_retries=2,
            )
            t = validate_accident_time(result or {})
            print_debug_payload("stage1", result)

            if t is not None and not is_near_zero(t, STAGE1_MIN_SEC):
                accident_time = snap_time_to_frame(
                    t, video_info["fps"], video_info["total_frames"]
                ) if video_info["fps"] > 0 else t
                retry_count = attempt
                break
            print(f"  [재시도] 0초 근처 (t={t}), 재시도 {attempt + 1}/{STAGE1_MAX_RETRIES}")

        if accident_time is None:
            accident_time = round(video_info.get("duration", 0) / 2, 3)
            print(f"  [fallback] {accident_time:.3f}s 사용")

        frame_index = round(accident_time * video_info["fps"]) if video_info["fps"] > 0 else 0
        print(f"  -> Stage1: {accident_time:.3f}s (frame={frame_index}, retry={retry_count})")

        append_stage_row(run_dir, 1, {
            "path": video_name, "accident_time": round(accident_time, 4),
            "frame_index": frame_index, "retry_count": retry_count,
        })
        print(f"  완료 ({format_elapsed(time.time() - t0)})")
