"""
video.py — 비디오 프레임 샘플링, 추출, 클립 저장.
load_clip_frames_for_vlm: 지정 구간을 고해상도/고프레임으로 샘플링 (clip Stage2용).
"""
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import STAGE1_MAX_FRAMES, STAGE2_CLIP_MAX_FRAMES, STAGE2_CLIP_MAX_PIXELS


def probe_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0, "duration": 0.0}
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = round(total_frames / fps, 3) if fps > 0 else 0.0
    return {"width": width, "height": height, "fps": round(fps, 3),
            "total_frames": total_frames, "duration": duration}


def load_video_frames_for_vlm(
    video_path: str,
    sample_fps: float,
    max_frames: int = STAGE1_MAX_FRAMES,
) -> Optional[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_fps      = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if raw_fps <= 0:
        raw_fps = max(sample_fps, 1.0)
    if total_frames <= 0:
        cap.release()
        return None

    step          = max(1, int(round(raw_fps / max(sample_fps, 0.1))))
    frame_indices = list(range(0, total_frames, step))
    if not frame_indices or frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    if len(frame_indices) > max_frames:
        scale   = (len(frame_indices) - 1) / max(max_frames - 1, 1)
        reduced = [frame_indices[min(round(i * scale), len(frame_indices) - 1)]
                   for i in range(max_frames)]
        frame_indices = sorted(set(reduced))

    frames: List[Image.Image] = []
    target_set     = set(frame_indices)
    next_targets   = iter(sorted(target_set))
    current_target = next(next_targets, None)
    current_index  = 0

    while current_target is not None:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if current_index == current_target:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            current_target = next(next_targets, None)
        current_index += 1

    cap.release()
    if not frames:
        return None

    timestamps = [frame_indices[i] / raw_fps for i in range(len(frames))]
    return {
        "frames"       : frames,
        "raw_fps"      : raw_fps,
        "sample_fps"   : sample_fps,
        "num_frames"   : len(frames),
        "frame_indices": frame_indices[:len(frames)],
        "timestamps"   : timestamps,
    }


def load_clip_frames_for_vlm(
    video_path: str,
    clip_start: float,
    clip_end: float,
    sample_fps: float = 10.0,
    max_frames: int = STAGE2_CLIP_MAX_FRAMES,
    max_pixels: int = STAGE2_CLIP_MAX_PIXELS,
) -> Optional[Dict[str, Any]]:
    """clip_start ~ clip_end 구간을 고해상도/고프레임으로 샘플링합니다 (Stage2 clip용)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_fps      = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if raw_fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    start_fi = max(0, int(clip_start * raw_fps))
    end_fi   = min(total_frames - 1, int(clip_end * raw_fps))
    if start_fi >= end_fi:
        cap.release()
        return None

    step          = max(1, int(round(raw_fps / max(sample_fps, 0.1))))
    frame_indices = list(range(start_fi, end_fi + 1, step))
    if not frame_indices or frame_indices[-1] != end_fi:
        frame_indices.append(end_fi)

    if len(frame_indices) > max_frames:
        scale   = (len(frame_indices) - 1) / max(max_frames - 1, 1)
        reduced = [frame_indices[min(round(i * scale), len(frame_indices) - 1)]
                   for i in range(max_frames)]
        frame_indices = sorted(set(reduced))

    frames: List[Image.Image] = []
    timestamps: List[float]   = []

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = resize_image_to_pixel_budget(pil, max_pixels)
        frames.append(pil)
        timestamps.append(round(fi / raw_fps, 3))

    cap.release()
    if not frames:
        return None

    return {"frames": frames, "timestamps": timestamps, "num_frames": len(frames)}


def resize_image_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale      = (max_pixels / float(width * height)) ** 0.5
    new_width  = max(28, (max(28, int(width  * scale)) // 28) * 28)
    new_height = max(28, (max(28, int(height * scale)) // 28) * 28)
    return image.resize((new_width, new_height), Image.LANCZOS)


def extract_frame_at_time(
    video_path: str, accident_time: float, fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps          = fps_override or cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        return None

    frame_index = max(0, min(int(accident_time * fps), max(0, total_frames - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    cap.release()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, accident_time) * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return {"frame": frame, "fps": fps, "frame_index": frame_index}
    return None


def save_frame_jpg(frame_bgr: np.ndarray, output_path: str) -> bool:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.imwrite(output_path, frame_bgr)


def extract_clip(video_path: str, output_path: str, start_sec: float, end_sec: float) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_sec * fps))
    end_frame   = min(total_frames, int(end_sec * fps))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0
