"""
pipeline.py — 4단계 파이프라인 오케스트레이션.

Stage 1 : 전체 영상 → accident_time 추론
Stage 2 : 전체 영상 + 약한 힌트 → 정밀 accident_time + 키프레임/클립 저장
Stage 3 : 4초 클립 → is_single + type 분류
Stage 4 : 키프레임 + Stage3 결과 → center_x, center_y
"""
import os
import time
from typing import Optional

from .config import OUTPUT_ROOT, VIDEO_DIR
from .stage1.full import run_stage1
from .stage2.full import run_stage2
from .stage3.run import run_stage3
from .stage4.run import run_stage4
from .utils.io_utils import build_video_lookup, get_run_dir, scan_video_files
from .utils.logging_utils import format_elapsed
from .utils.model import load_vllm_model


def main(
    run_id: str,
    from_stage: int = 1,
    only_stage: Optional[int] = None,
    video_dir: str = VIDEO_DIR,
    limit: Optional[int] = None,
) -> None:
    run_dir = get_run_dir(OUTPUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    video_files  = scan_video_files(video_dir)
    if limit:
        video_files = video_files[:limit]
    video_lookup = build_video_lookup(video_dir)

    print(f"\n{'='*60}")
    print(f"Run ID   : {run_id}")
    print(f"Run Dir  : {run_dir}")
    print(f"Videos   : {len(video_files)}개")
    print(f"From     : Stage {from_stage}" + (f" (only Stage {only_stage})" if only_stage else ""))
    print(f"{'='*60}")

    t_load = time.time()
    model, sampling_params, stage1_sampling_params = load_vllm_model()
    print(f"\n모델 로드 완료 ({format_elapsed(time.time() - t_load)})")

    stages_to_run = [only_stage] if only_stage else list(range(from_stage, 5))
    t_total = time.time()

    for stage_num in stages_to_run:
        if stage_num == 1:
            run_stage1(model, stage1_sampling_params, run_dir, video_files, video_lookup)
        elif stage_num == 2:
            run_stage2(model, sampling_params, run_dir, video_lookup)
        elif stage_num == 3:
            run_stage3(model, sampling_params, run_dir, video_lookup)
        elif stage_num == 4:
            run_stage4(model, sampling_params, run_dir, video_lookup)

    print(f"\n{'='*60}")
    print(f"완료 — 총 소요시간: {format_elapsed(time.time() - t_total)}")
    print(f"결과: {run_dir}")
    print(f"{'='*60}")
