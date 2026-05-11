"""
run_reproduce.py — full+clip min_ensemble + Stage3/4

흐름:
  1. [full 방식] Stage 1 시간 추론 흐름  →  {run_id}_full/stage2.csv
  2. [clip 방식] Stage 1 시간 추론 흐름  →  {run_id}_clip/stage2.csv
  3. min_ensemble(stage2)   →  {run_id}/stage2.csv
  4. Stage 3/4              →  {run_id}/stage3.csv, stage4.csv, submission.csv

사용법:
  python src/run_reproduce.py --run-id reproduce_v1
  python src/run_reproduce.py --run-id reproduce_test --limit 10
  python src/run_reproduce.py --run-id reproduce_v1 --from-step 2
  python src/run_reproduce.py --run-id reproduce_v1 --from-step 3
  python src/run_reproduce.py --run-id reproduce_v1 --from-step 4
"""
import argparse
import os
import sys
import subprocess
import time
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
sys.path.insert(0, BASE)
os.environ.setdefault("PIPELINE_VARIANT", "full_run")

from src.config import OUTPUT_ROOT, VIDEO_DIR


def run_subprocess(script: str, run_id: str, video_dir: str,
                   limit: int = None, extra_args: list = None) -> None:
    cmd = [sys.executable, os.path.join(BASE, "src", script),
           "--run-id", run_id, "--video-dir", video_dir]
    if limit:
        cmd += ["--limit", str(limit)]
    if extra_args:
        cmd += extra_args
    print(f"  실행: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=BASE)
    if ret.returncode != 0:
        print(f"[오류] {script} 실패 (returncode={ret.returncode})")
        sys.exit(1)


def min_ensemble_stage2(run_dir_full: str, run_dir_clip: str, run_dir_out: str,
                        limit: int = None) -> bool:
    """full/stage2 + clip/stage2 → accident_time 최솟값 → {run_id}/stage2.csv"""
    full_csv = os.path.join(run_dir_full, "stage2.csv")
    clip_csv = os.path.join(run_dir_clip, "stage2.csv")
    out_csv  = os.path.join(run_dir_out,  "stage2.csv")

    if not os.path.exists(full_csv):
        print(f"[오류] {full_csv} 없음")
        return False
    if not os.path.exists(clip_csv):
        print(f"[오류] {clip_csv} 없음")
        return False

    full = pd.read_csv(full_csv)
    clip = pd.read_csv(clip_csv)

    if limit:
        full = full.head(limit)
        clip = clip.head(limit)

    full["path_key"] = full["path"].str.replace("videos/", "", regex=False)
    clip["path_key"] = clip["path"].str.replace("videos/", "", regex=False)

    merged = full.merge(clip[["path_key", "accident_time"]], on="path_key", suffixes=("", "_clip"))
    merged["accident_time"] = merged[["accident_time", "accident_time_clip"]].min(axis=1)
    merged = merged.drop(columns=["accident_time_clip", "path_key"])

    os.makedirs(run_dir_out, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    changed = (full.set_index("path")["accident_time"] != merged.set_index("path")["accident_time"]).sum()
    print(f"  완료: {out_csv}")
    print(f"    총 영상  : {len(merged)}개")
    print(f"    시간 변경: {changed}개 ({changed/len(merged)*100:.1f}%)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="full+clip min_ensemble + Stage3/4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id",    type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3, 4])
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id or time.strftime("reproduce_%Y%m%d_%H%M%S")

    run_dir_full = os.path.join(OUTPUT_ROOT, f"{run_id}_full")
    run_dir_clip = os.path.join(OUTPUT_ROOT, f"{run_id}_clip")
    run_dir_out  = os.path.join(OUTPUT_ROOT, run_id)

    print(f"\n{'='*70}")
    print(f"  full+clip min_ensemble + Stage3/4")
    print(f"{'='*70}")
    print(f"  Run ID    : {run_id}")
    print(f"  Videos    : {args.limit or '전체'}")
    print(f"  From Step : {args.from_step}")
    print(f"{'='*70}\n")

    t0 = time.time()

    if args.from_step <= 1:
        print(f"{'─'*70}")
        print(f"[Step 1] full Stage 1 time flow  →  {run_id}_full/")
        print(f"{'─'*70}")
        run_subprocess("run.py", f"{run_id}_full", args.video_dir, args.limit,
                       extra_args=["--from-stage", "1", "--only-stage", "1"])

    if args.from_step <= 2:
        print(f"\n{'─'*70}")
        print(f"[Step 2] clip Stage 1 time flow  →  {run_id}_clip/")
        print(f"{'─'*70}")
        run_subprocess("run_clip.py", f"{run_id}_clip", args.video_dir, args.limit,
                       extra_args=["--from-stage", "1", "--only-stage", "1"])

    if args.from_step <= 3:
        print(f"\n{'─'*70}")
        print(f"[Step 3] min_ensemble  →  {run_id}/stage2.csv")
        print(f"{'─'*70}")
        if not min_ensemble_stage2(run_dir_full, run_dir_clip, run_dir_out, limit=args.limit):
            sys.exit(1)

    print(f"\n{'─'*70}")
    print(f"[Step 4] Stage 3/4  →  {run_id}/submission.csv")
    print(f"{'─'*70}")
    run_subprocess("run.py", run_id, args.video_dir, args.limit,
                   extra_args=["--from-stage", "3"])

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n{'='*70}")
    print(f"  완료 — 총 소요시간: {h}시간 {m}분 {s}초")
    print(f"  결과: output/{run_id}/submission.csv")
    print(f"{'='*70}")
