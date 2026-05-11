"""
stage3/run.py — 사고 유형 분류.
"""
import time
from typing import Any, Dict, Optional

from ..config import (
    DEBUG_VISIBLE_EVIDENCE,
    SKIP_EXISTING, STAGE3_MAX_FRAMES, STAGE3_TARGET_FPS,
)
from ..utils.inference import call_qwen_for_video
from ..utils.io_utils import (
    append_stage_row, get_processed_paths, load_stage_csv, resolve_video_path,
)
from ..utils.logging_utils import format_elapsed
from ..utils.validators import (
    print_debug_payload, validate_is_single, validate_type_prediction,
)
from ..utils.video import probe_video_info


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


def build_stage3_prompt(
    accident_time: float,
    video_info: Optional[Dict[str, Any]] = None,
) -> str:
    if DEBUG_VISIBLE_EVIDENCE:
        output_format = ('Output format:\n{\n  "is_single": <true | false>,\n'
                         '  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>",\n'
                         '  "confidence": <float 0.0-1.0>,\n  "evidence": "<brief cues>"\n}')
    else:
        output_format = ('Output format:\n{\n  "is_single": <true | false>,\n'
                         '  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"\n}')

    return f"""You are an expert traffic accident analyst.
You are given a short video clip (approximately 4 seconds) around the accident moment at {accident_time:.3f}s.
{_video_props(video_info)}
Step 1 — Determine is_single:
- true  : Only ONE vehicle is involved (hit a pole, barrier, guardrail, rollover, etc.)
- false : TWO or more vehicles are involved

Step 2 — If is_single is false, classify the type:
- rear-end  : One vehicle crashes into the back of another traveling in the same direction
- head-on   : Two vehicles traveling in opposite directions collide front-to-front
- sideswipe : Two vehicles in roughly the same direction make side-to-side contact
- t-bone    : Front of one vehicle hits the side of another (T shape)

If is_single is true, set type = "single".

Instructions:
1. First decide is_single by examining all visible vehicles.
2. Then classify the type based on vehicle approach angles and contact region.
3. Choose exactly ONE type from: [rear-end, head-on, sideswipe, t-bone, single].

Critical output rules:
- Output exactly ONE JSON object.
- No markdown, no code fences, no text before or after JSON.
{output_format}""".strip()


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def run_stage3(
    model, sampling_params, run_dir: str, video_lookup: Dict[str, str],
) -> None:
    import os
    stage2_data = load_stage_csv(run_dir, 2)
    if not stage2_data:
        print("[Stage 3] stage2.csv 없음 — Stage 2를 먼저 실행하세요.")
        return

    print(f"\n[Stage 3] 사고 유형 분류 — {len(stage2_data)}개 영상")
    processed = get_processed_paths(run_dir, 3) if SKIP_EXISTING else set()

    for video_name, s2_row in stage2_data.items():
        if SKIP_EXISTING and video_name in processed:
            print(f"  [SKIP] {video_name}")
            continue

        t0            = time.time()
        accident_time = float(s2_row.get("accident_time") or 0)
        clip_path     = s2_row.get("clip_path", "")
        abs_path      = resolve_video_path(video_name, video_lookup)

        print(f"\n  처리: {video_name}  (accident_time={accident_time:.3f}s)")

        video_source = clip_path if clip_path and os.path.exists(clip_path) else abs_path
        if not video_source:
            print(f"  [오류] 영상/클립 없음: {video_name}")
            append_stage_row(run_dir, 3, {
                "path": video_name, "type": "single", "is_single": 1, "evidence": "fallback",
            })
            continue

        video_info = probe_video_info(abs_path) if abs_path else {}
        prompt     = build_stage3_prompt(accident_time, video_info=video_info)
        result     = call_qwen_for_video(
            model, sampling_params, video_source, prompt,
            label=f"stage3 {video_name}",
            sample_fps=STAGE3_TARGET_FPS, max_frames=STAGE3_MAX_FRAMES,
        )
        print_debug_payload("stage3", result)

        is_single = validate_is_single(result or {})
        acc_type  = validate_type_prediction(result or {})
        evidence  = (result or {}).get("evidence", "")

        if is_single is True:
            acc_type = "single"
        elif acc_type is None:
            acc_type = "single"
        if is_single is None:
            is_single = (acc_type == "single")

        print(f"  -> Stage3: type={acc_type}, is_single={is_single}")

        append_stage_row(run_dir, 3, {
            "path"     : video_name,
            "type"     : acc_type,
            "is_single": int(is_single),
            "evidence" : str(evidence)[:200],
        })
        print(f"  완료 ({format_elapsed(time.time() - t0)})")
