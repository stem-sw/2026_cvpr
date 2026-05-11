"""
run.py — full_run variant 실행 진입점.

사용법:
  python src/run.py
  python src/run.py --run-id exp_v1
  python src/run.py --run-id exp_v1 --only-stage 1  # stage1.csv + stage2.csv
  python src/run.py --run-id exp_v1 --from-stage 3
  python src/run.py --run-id exp_v1 --only-stage 4
  python src/run.py --video-dir /path/to/videos
"""
import argparse
import os
import sys
import time

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PIPELINE_VARIANT", "full_run")

from src.config import VIDEO_DIR
from src.pipeline import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Accident Analysis Pipeline (full_run variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id",     type=str, default=None)
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--only-stage", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--video-dir",  type=str, default=VIDEO_DIR)
    parser.add_argument("--limit",      type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    main(
        run_id=run_id,
        from_stage=args.from_stage,
        only_stage=args.only_stage,
        video_dir=args.video_dir,
        limit=args.limit,
    )
