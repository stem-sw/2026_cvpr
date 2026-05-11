"""
run_tokens.py — tokens 실험: full+clip min_ensemble + 고해상도 Stage 3/4

흐름:
  1. [full variant] Stage 1 time flow  → output/{run_id}_full/
  2. [clip variant] Stage 1 time flow  → output/{run_id}_clip/
  3. min_ensemble (accident_time 최솟값)
  4. [tokens 설정] Stage 3/4   → output/{run_id}/

설정 (config_tokens.py):
  Stage 1:   fps=2.0, max_frames=120
  Stage 2:   fps=6.0, max_frames=120  (full: 전체 영상 / clip: ±2초 클립)
  Stage 3:   fps=10.0, max_frames=40
  Stage 4:   MAX_PIXELS=512×28²

사용법:
  python src/run_tokens.py --run-id tokens_v2
  python src/run_tokens.py --run-id tokens_test --limit 10
"""
import argparse
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIPELINE_VARIANT"] = "tokens"

from src.config import VIDEO_DIR, OUTPUT_ROOT
from src.utils.model import load_vllm_model
from src.stage1.full import run_stage1_flow
from src.stage1.clip import run_stage1_flow as run_stage1_clip_flow
from src.stage3.run import run_stage3
from src.stage4.run import run_stage4
from src.utils.io_utils import build_video_lookup, scan_video_files
from src.utils.logging_utils import format_elapsed


def min_ensemble(run_dir_full: str, run_dir_clip: str, run_dir_out: str, limit: int = None) -> bool:
    """full/stage2.csv + clip/stage2.csv → accident_time 최솟값 선택."""
    full_csv = os.path.join(run_dir_full, "stage2.csv")
    clip_csv = os.path.join(run_dir_clip, "stage2.csv")
    out_csv  = os.path.join(run_dir_out, "stage2.csv")

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
    merged.to_csv(out_csv, index=False)

    changed = (full.set_index("path")["accident_time"] != merged.set_index("path")["accident_time"]).sum()
    print(f"  min_ensemble 완료: {out_csv}")
    print(f"    총 영상: {len(merged)}개")
    print(f"    accident_time 변경: {changed}개 ({changed/len(merged)*100:.1f}%)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="tokens: full+clip min_ensemble + 고해상도 Stage 3/4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id",    type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--limit",     type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id or time.strftime("tokens_%Y%m%d_%H%M%S")

    run_dir_full = os.path.join(OUTPUT_ROOT, f"{run_id}_full")
    run_dir_clip = os.path.join(OUTPUT_ROOT, f"{run_id}_clip")
    run_dir_out  = os.path.join(OUTPUT_ROOT, run_id)

    for d in [run_dir_full, run_dir_clip, run_dir_out]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  tokens 실험: full+clip min_ensemble + 고해상도 Stage 3/4")
    print(f"{'='*70}")
    print(f"  Run ID  : {run_id}")
    print(f"  Videos  : {args.limit or '전체'}")
    print(f"  Stage 1 : fps=2.0, max_frames=120")
    print(f"  Stage 2 : fps=6.0, max_frames=120 (full) / clip±2s (clip)")
    print(f"  Stage 3 : fps=10.0, max_frames=40, MAX_PIXELS=512×28²")
    print(f"{'='*70}\n")

    print("[모델 로드]")
    t_load = time.time()
    model, sampling_params, stage1_sampling_params = load_vllm_model()
    print(f"  완료 ({format_elapsed(time.time() - t_load)})\n")

    video_files  = scan_video_files(args.video_dir)
    if args.limit:
        video_files = video_files[:args.limit]
    video_lookup = build_video_lookup(args.video_dir)

    # ── Step 1: full variant Stage 1 time flow ──────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 1] full variant Stage 1 time flow  →  {run_id}_full")
    print(f"{'─'*70}")
    run_stage1_flow(model, stage1_sampling_params, sampling_params,
                    run_dir_full, video_files, video_lookup)

    # ── Step 2: clip variant Stage 1 time flow ──────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 2] clip variant Stage 1 time flow  →  {run_id}_clip")
    print(f"{'─'*70}")
    run_stage1_clip_flow(model, stage1_sampling_params, sampling_params,
                         run_dir_clip, video_files, video_lookup)

    # ── Step 3: min_ensemble ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 3] min_ensemble  →  {run_id}/stage2.csv")
    print(f"{'─'*70}")
    if not min_ensemble(run_dir_full, run_dir_clip, run_dir_out, limit=args.limit):
        print("[오류] min_ensemble 실패")
        sys.exit(1)

    # ── Step 4: Stage 3/4 ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 4] Stage 3/4 고해상도  →  {run_id}")
    print(f"{'─'*70}")
    run_stage3(model, sampling_params, run_dir_out, video_lookup)
    run_stage4(model, sampling_params, run_dir_out, video_lookup)

    print(f"\n{'='*70}")
    print(f"  완료: output/{run_id}/submission.csv")
    print(f"{'='*70}")
