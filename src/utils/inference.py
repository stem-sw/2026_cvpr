"""
inference.py — vLLM 요청 빌드 및 실행.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..config import DEBUG_VISIBLE_EVIDENCE, MAX_CALL_RETRIES, TIME_STAGE_MAX_PIXELS
from .json_utils import extract_first_json_object
from .video import load_video_frames_for_vlm, resize_image_to_pixel_budget


def render_chatml_prompt(messages: List[Dict[str, Any]]) -> Tuple[str, List[Image.Image]]:
    parts: List[str]          = []
    images: List[Image.Image] = []

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        role_parts: List[str] = []

        if isinstance(content, str):
            role_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    role_parts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    img = item.get("image")
                    if not isinstance(img, Image.Image):
                        raise TypeError(f"Expected PIL.Image, got {type(img)}")
                    images.append(img)
                    role_parts.append("<|vision_start|><|image_pad|><|vision_end|>")

        parts.append(f"<|im_start|>{role}\n{''.join(role_parts)}<|im_end|>")

    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n")
    return "\n".join(parts), images


def messages_to_vllm_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt, images = render_chatml_prompt(messages)
    req: Dict[str, Any] = {"prompt": prompt}
    if images:
        req["multi_modal_data"] = {"image": images}
    return req


def _system_text() -> str:
    return "Respond with exactly one JSON object. No markdown. No code fences. No explanation."


def _summarize_result(result: Dict[str, Any]) -> str:
    ordered_keys = [
        "accident_time", "type", "is_single", "center_x", "center_y",
        "confidence", "candidates",
    ]
    parts = []
    for key in ordered_keys:
        if key not in result:
            continue
        value = result[key]
        if key == "candidates" and isinstance(value, list):
            parts.append(f"candidates={len(value)}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else f"keys={sorted(result.keys())}"


def run_vllm_request(
    model, sampling_params,
    messages: List[Dict[str, Any]],
    label: str = "",
    max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    request      = messages_to_vllm_request(messages)
    last_error   = None

    for attempt in range(1, max_retries + 1):
        try:
            outputs = model.generate([request], sampling_params=sampling_params)
            collected_text = ""
            if outputs and outputs[0].outputs:
                collected_text = outputs[0].outputs[0].text.strip()

            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                print(f"  -> [{label}] 시도 {attempt}/{max_retries}: {_summarize_result(parsed)}")
                if DEBUG_VISIBLE_EVIDENCE:
                    print(f"     raw={collected_text[:240]}")
                return parsed
            last_error = f"JSON 파싱 실패 (시도 {attempt})"
            print(f"  -> [{label}] 시도 {attempt}/{max_retries}: JSON 파싱 실패")
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)

    print(f"    [오류] [{label}] 최종 실패: {last_error}")
    return None


def call_qwen_for_frame_sequence(
    model, sampling_params,
    frames: List[Any], timestamps: List[float],
    prompt: str, label: str = "", max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    resized = [resize_image_to_pixel_budget(f, TIME_STAGE_MAX_PIXELS) for f in frames]
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, (frame, ts) in enumerate(zip(resized, timestamps)):
        content.append({"type": "text",  "text": f"Frame {idx} at {ts:.3f}s."})
        content.append({"type": "image", "image": frame})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _system_text()}]},
        {"role": "user",   "content": content},
    ]
    return run_vllm_request(model, sampling_params, messages, label, max_retries)


def call_qwen_for_video(
    model, sampling_params,
    video_path: str, prompt: str,
    label: str = "", sample_fps: float = 6.0, max_frames: int = 32,
    max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    sampled = load_video_frames_for_vlm(video_path, sample_fps=sample_fps, max_frames=max_frames)
    if not sampled:
        print(f"  -> [{label}] 비디오 샘플링 실패: {video_path}")
        return None
    print(f"  -> [{label}] 샘플링: {sampled['num_frames']}프레임 (fps={sample_fps})")
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, (frame, ts) in enumerate(zip(sampled["frames"], sampled["timestamps"])):
        content.append({"type": "text",  "text": f"Frame {idx} at {ts:.3f}s."})
        content.append({"type": "image", "image": frame})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _system_text()}]},
        {"role": "user",   "content": content},
    ]
    return run_vllm_request(model, sampling_params, messages, label, max_retries)


def call_qwen_for_image(
    model, sampling_params,
    image_path: str, prompt: str,
    label: str = "", max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    with Image.open(image_path) as img:
        pil_image = img.convert("RGB")
    content = [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text":  prompt},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _system_text()}]},
        {"role": "user",   "content": content},
    ]
    return run_vllm_request(model, sampling_params, messages, label, max_retries)
