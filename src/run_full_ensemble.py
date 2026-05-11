"""
run_full_ensemble.py — full_run config(stage1 196프레임) + full/clip min_ensemble + Stage 3/4

흐름:
  1. [full variant]  Stage 1/2  → output/{run_id}_full/
  2. [clip variant]  Stage 1/2  → output/{run_id}_clip/
  3. min_ensemble (accident_time 최솟값)  → output/{run_id}/stage2.csv
  4. Stage 3/4                           → output/{run_id}/

설정 (config.py, PIPELINE_VARIANT=full_run):
  Stage 1:  fps=2.0,  max_frames=196
  Stage 2:  fps=6.0,  max_frames=120  (full: 전체 영상 / clip: ±2초 클립)
  Stage 3:  fps=6.0,  max_frames=32
  해상도:   MAX_PIXELS=1024×28²

tokens_v2와의 차이:
  - stage1 max_frames: 196 (이 실험) vs 120 (tokens_v2)
  - stage3 fps/frames: 6.0/32 (이 실험) vs 10.0/40 (tokens_v2)
  - MAX_PIXELS: 1024×28² (이 실험) vs 512×28² (tokens_v2)

사용법:
  python src/run_full_ensemble.py --run-id full_ens_v1
  python src/run_full_ensemble.py --run-id full_ens_test --limit 10
"""
import argparse
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIPELINE_VARIANT"] = "full_run"

from src.config import VIDEO_DIR, OUTPUT_ROOT
from src.utils.model import load_vllm_model
from src.stage1.full import run_stage1
from src.stage2.full import run_stage2
from src.stage2.clip import run_stage2 as run_stage2_clip
from src.stage3.run import run_stage3
from src.stage4.run import run_stage4
from src.utils.io_utils import build_video_lookup, scan_video_files
from src.utils.logging_utils import format_elapsed


def min_ensemble(run_dir_full: str, run_dir_clip: str, run_dir_out: str, limit: int = None) -> bool:
    """full/stage2.csv + clip/stage2.csv → accident_time 최솟값 기준으로 승자 행 전체 선택."""
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

    merged = full.merge(clip, on="path_key", suffixes=("_full", "_clip"))

    # clip이 이기면 clip 행 전체(accident_time, clip_path, key_frame_path 등) 채택
    clip_wins = merged["accident_time_clip"] < merged["accident_time_full"]
    row_cols = [c for c in full.columns if c != "path_key"]
    for col in row_cols:
        full_col = f"{col}_full"
        clip_col = f"{col}_clip"
        if full_col in merged.columns and clip_col in merged.columns:
            merged[col] = merged[clip_col].where(clip_wins, merged[full_col])

    merged = merged[["path"] + [c for c in row_cols if c != "path"]].copy()
    merged.to_csv(out_csv, index=False)

    changed = clip_wins.sum()
    print(f"  min_ensemble 완료: {out_csv}")
    print(f"    총 영상: {len(merged)}개")
    print(f"    clip 승리(행 전체 교체): {changed}개 ({changed/len(merged)*100:.1f}%)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="full_run config + full/clip min_ensemble + Stage 3/4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id",    type=str, default=None, help="실험 ID (기본: full_ens_YYYYMMDD_HHMMSS)")
    parser.add_argument("--video-dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--limit",     type=int, default=None, help="최대 영상 수 (테스트: --limit 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id or time.strftime("full_ens_%Y%m%d_%H%M%S")

    run_dir_full = os.path.join(OUTPUT_ROOT, f"{run_id}_full")
    run_dir_clip = os.path.join(OUTPUT_ROOT, f"{run_id}_clip")
    run_dir_out  = os.path.join(OUTPUT_ROOT, run_id)

    for d in [run_dir_full, run_dir_clip, run_dir_out]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  full_run config + full/clip min_ensemble + Stage 3/4")
    print(f"{'='*70}")
    print(f"  Run ID  : {run_id}")
    print(f"  Videos  : {args.limit or '전체'}")
    print(f"  Sub-runs: {run_id}_full / {run_id}_clip")
    print(f"  Stage 1 : fps=2.0, max_frames=196 (full) / 120 (clip)")
    print(f"  Stage 2 : fps=6.0, max_frames=120 (full) / clip±2s (clip)")
    print(f"  Stage 3 : fps=6.0, max_frames=32, MAX_PIXELS=1024×28²")
    print(f"{'='*70}\n")

    print("[모델 로드]")
    t_load = time.time()
    model, sampling_params, stage1_sampling_params = load_vllm_model()
    print(f"  완료 ({format_elapsed(time.time() - t_load)})\n")

    video_files  = scan_video_files(args.video_dir)
    if args.limit:
        video_files = video_files[:args.limit]
    video_lookup = build_video_lookup(args.video_dir)

    # ── Step 1: full variant Stage 1/2 ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 1] full variant Stage 1/2  →  {run_id}_full")
    print(f"{'─'*70}")
    run_stage1(model, stage1_sampling_params, run_dir_full, video_files, video_lookup)
    run_stage2(model, sampling_params, run_dir_full, video_lookup)

    # ── Step 2: clip variant Stage 1/2 ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 2] clip variant Stage 1/2  →  {run_id}_clip")
    print(f"{'─'*70}")
    run_stage1(model, stage1_sampling_params, run_dir_clip, video_files, video_lookup, max_frames=120)
    run_stage2_clip(model, sampling_params, run_dir_clip, video_lookup)

    # ── Step 3: min_ensemble ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 3] min_ensemble  →  {run_id}/stage2.csv")
    print(f"{'─'*70}")
    if not min_ensemble(run_dir_full, run_dir_clip, run_dir_out, limit=args.limit):
        print("[오류] min_ensemble 실패")
        sys.exit(1)

    # ── Step 4: Stage 3/4 ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[Step 4] Stage 3/4  →  {run_id}")
    print(f"{'─'*70}")
    run_stage3(model, sampling_params, run_dir_out, video_lookup)
    run_stage4(model, sampling_params, run_dir_out, video_lookup)

    print(f"\n{'='*70}")
    print(f"  완료: output/{run_id}/submission.csv")
    print(f"{'='*70}")
