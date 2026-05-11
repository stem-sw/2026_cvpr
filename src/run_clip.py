"""
run_clip.py — clip variant 실행 진입점.

Stage 1 time flow에서 Top-3 후보를 추출한 뒤,
각 후보 ±2초 클립을 개별 분석하여 신뢰도 최고 결과를 사용합니다.

사용법:
  python src/run_clip.py --run-id exp_clip_v1
  python src/run_clip.py --run-id exp_clip_v1 --only-stage 1
  python src/run_clip.py --run-id exp_clip_v1 --from-stage 3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PIPELINE_VARIANT", "clip")

from src.config import VIDEO_DIR
from src.pipeline_clip import main


def parse_args():
    parser = argparse.ArgumentParser(description="Clip Focus Pipeline Variant")
    parser.add_argument("--run-id",     type=str, default=None)
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--only-stage", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--video-dir",  type=str, default=VIDEO_DIR)
    parser.add_argument("--limit",      type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id or time.strftime("clip_%Y%m%d_%H%M%S")
    main(
        run_id=run_id,
        from_stage=args.from_stage,
        only_stage=args.only_stage,
        video_dir=args.video_dir,
        limit=args.limit,
    )
