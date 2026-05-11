"""
validators.py — 예측 결과 검증, snap_time_to_frame.
"""
import re
from typing import Any, Dict, Optional

from ..config import DEBUG_VISIBLE_EVIDENCE, MAX_EVIDENCE_CHARS, VALID_TYPES


def snap_time_to_frame(accident_time: float, fps: float, total_frames: int) -> float:
    if fps <= 0:
        return accident_time
    frame_idx = round(accident_time * fps)
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    return round(frame_idx / fps, 4)


def is_near_zero(t: Optional[float], min_sec: float) -> bool:
    return t is None or t < min_sec


def validate_accident_time(result: Dict[str, Any]) -> Optional[float]:
    try:
        return float(result["accident_time"])
    except (KeyError, TypeError, ValueError):
        return None


def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        return {
            "center_x": min(max(float(result["center_x"]), 0.0), 1.0),
            "center_y": min(max(float(result["center_y"]), 0.0), 1.0),
        }
    except (KeyError, TypeError, ValueError):
        return None


def validate_type_prediction(result: Dict[str, Any]) -> Optional[str]:
    try:
        t = str(result["type"]).strip().lower()
        return t if t in VALID_TYPES else None
    except (KeyError, TypeError, ValueError):
        return None


def validate_is_single(result: Dict[str, Any]) -> Optional[bool]:
    raw = result.get("is_single")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1")
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        return None


def maybe_get_confidence(result: Optional[Dict[str, Any]]) -> Optional[float]:
    if result is None:
        return None
    try:
        return min(max(float(result["confidence"]), 0.0), 1.0)
    except (KeyError, TypeError, ValueError):
        return None


def maybe_get_evidence(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if result is None:
        return None
    raw = result.get("evidence")
    if not raw:
        return None
    evidence = re.sub(r"\s+", " ", str(raw).strip())
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS - 3].rstrip() + "..."
    return evidence or None


def print_debug_payload(stage: str, result: Optional[Dict[str, Any]]) -> None:
    if not DEBUG_VISIBLE_EVIDENCE or result is None:
        return
    parts = []
    conf = maybe_get_confidence(result)
    evid = maybe_get_evidence(result)
    if conf is not None:
        parts.append(f"confidence={conf:.2f}")
    if evid:
        parts.append(f"evidence={evid}")
    if parts:
        print(f"  -> [{stage}] " + " | ".join(parts))
