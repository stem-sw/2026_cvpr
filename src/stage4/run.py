"""
stage4/run.py — 충돌 위치 추론.
"""
import os
import time
from typing import Any, Dict, Optional

from ..config import DEBUG_VISIBLE_EVIDENCE, SKIP_EXISTING
from ..utils.inference import call_qwen_for_image
from ..utils.io_utils import (
    append_stage_row, append_submission_row, get_artifacts_dir,
    get_processed_paths, load_stage_csv, resolve_video_path,
)
from ..utils.logging_utils import format_elapsed
from ..utils.validators import print_debug_payload, validate_location_prediction
from ..utils.video import extract_frame_at_time, probe_video_info, save_frame_jpg


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

def build_stage4_prompt(
    accident_time: float,
    accident_type: str,
    is_single: bool,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    lines = []
    if video_info:
        w, h = video_info.get("width", 0), video_info.get("height", 0)
        fps  = video_info.get("fps", 0.0)
        dur  = video_info.get("duration", 0.0)
        if w and h: lines.append(f"- resolution: {w}x{h}")
        if fps:     lines.append(f"- fps: {fps:.3f}")
        if dur:     lines.append(f"- duration: {dur:.3f}s")
    video_section = ("\nVideo properties:\n" + "\n".join(lines) + "\n") if lines else ""

    scope = "single-vehicle" if is_single else "multi-vehicle"

    type_hint_map = {
        "rear-end" : "The accident is a rear-end collision — focus on the rear of the leading vehicle.",
        "head-on"  : "The accident is a head-on collision — focus on the front contact region between two vehicles.",
        "sideswipe": "The accident is a sideswipe — focus on the side-to-side contact area.",
        "t-bone"   : "The accident is a T-bone — focus on the side of the struck vehicle where the front hits.",
        "single"   : "This is a single-vehicle accident — focus on where the vehicle contacts the obstacle.",
    }
    type_hint = type_hint_map.get(accident_type, "")

    if DEBUG_VISIBLE_EVIDENCE:
        output_format = ('Output format:\n{\n  "center_x": <float 0.0-1.0, left=0 right=1>,\n'
                         '  "center_y": <float 0.0-1.0, top=0 bottom=1>,\n'
                         '  "confidence": <float 0.0-1.0>,\n  "evidence": "<brief visible cues>"\n}')
    else:
        output_format = ('Output format:\n{\n  "center_x": <float 0.0-1.0, left=0 right=1>,\n'
                         '  "center_y": <float 0.0-1.0, top=0 bottom=1>\n}')

    return f"""You are an expert traffic accident analyst.
You are given a single key frame from CCTV footage at accident_time = {accident_time:.3f}s.
{video_section}
Prior step result:
- accident_scope : {scope}
- predicted_type : {accident_type}
{type_hint}

Task: Precisely localize the PRIMARY collision point in this frame.

Instructions:
1. Focus on the main area where physical contact occurs.
2. Output normalized coordinates of the CENTER of the collision region:
   - center_x: 0.0 = left edge, 1.0 = right edge
   - center_y: 0.0 = top edge,  1.0 = bottom edge
3. Target the actual contact point, NOT the center of the whole vehicle.
4. Use the accident type hint above to guide where to look.
5. If uncertain, output a single best estimate.

Critical output rules:
- Output exactly ONE JSON object.
- No markdown, no code fences, no text before or after JSON.
{output_format}""".strip()


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------

def run_stage4(
    model, sampling_params, run_dir: str, video_lookup: Dict[str, str],
) -> None:
    stage2_data = load_stage_csv(run_dir, 2)
    stage3_data = load_stage_csv(run_dir, 3)

    if not stage2_data:
        print("[Stage 4] stage2.csv 없음 — Stage 1 time flow를 먼저 실행하세요.")
        return

    print(f"\n[Stage 4] 충돌 위치 추론 — {len(stage2_data)}개 영상")
    processed = get_processed_paths(run_dir, 4) if SKIP_EXISTING else set()

    for video_name, s2_row in stage2_data.items():
        if SKIP_EXISTING and video_name in processed:
            print(f"  [SKIP] {video_name}")
            continue

        t0             = time.time()
        accident_time  = float(s2_row.get("accident_time") or 0)
        key_frame_path = s2_row.get("key_frame_path", "")
        abs_path       = resolve_video_path(video_name, video_lookup)

        s3_row    = stage3_data.get(video_name, {})
        acc_type  = s3_row.get("type", "single")
        is_single = bool(int(s3_row.get("is_single", 1)))

        print(f"\n  처리: {video_name}  (time={accident_time:.3f}s, type={acc_type})")

        if not key_frame_path or not os.path.exists(key_frame_path):
            if abs_path:
                stem           = os.path.splitext(video_name)[0]
                artifact_dir   = get_artifacts_dir(run_dir, stem)
                key_frame_path = os.path.join(artifact_dir, "key_frame.jpg")
                frame_result   = extract_frame_at_time(abs_path, accident_time)
                if frame_result:
                    save_frame_jpg(frame_result["frame"], key_frame_path)
                else:
                    key_frame_path = ""

        if not key_frame_path or not os.path.exists(key_frame_path):
            print(f"  [오류] 키프레임 없음: {video_name}")
            append_stage_row(run_dir, 4, {"path": video_name, "center_x": 0.5, "center_y": 0.5})
            append_submission_row(run_dir, {
                "path": video_name, "accident_time": round(accident_time, 4),
                "center_x": 0.5, "center_y": 0.5, "type": acc_type,
            })
            continue

        video_info = probe_video_info(abs_path) if abs_path else {}
        prompt     = build_stage4_prompt(accident_time, acc_type, is_single, video_info=video_info)
        result     = call_qwen_for_image(
            model, sampling_params, key_frame_path, prompt,
            label=f"stage4 {video_name}",
        )
        print_debug_payload("stage4", result)

        loc = validate_location_prediction(result or {})
        if loc is None:
            loc = {"center_x": 0.5, "center_y": 0.5}

        print(f"  -> Stage4: center_x={loc['center_x']:.4f}, center_y={loc['center_y']:.4f}")

        append_stage_row(run_dir, 4, {
            "path"    : video_name,
            "center_x": round(loc["center_x"], 6),
            "center_y": round(loc["center_y"], 6),
        })
        append_submission_row(run_dir, {
            "path"         : video_name,
            "accident_time": round(accident_time, 4),
            "center_x"     : round(loc["center_x"], 6),
            "center_y"     : round(loc["center_y"], 6),
            "type"         : acc_type,
        })
        print(f"  완료 ({format_elapsed(time.time() - t0)})")
