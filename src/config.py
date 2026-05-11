"""
config.py — 재현 파이프라인 전체 설정값.

PIPELINE_VARIANT 환경변수에 따라 variant별 값이 결정됩니다.
  full_run  →  STAGE1_MAX_FRAMES=196, MAX_PIXELS=1024×28²  (기본값)
  clip      →  STAGE1_MAX_FRAMES=120, MAX_PIXELS=1024×28²
  tokens    →  STAGE1_MAX_FRAMES=120, STAGE3 fps/frames↑, MAX_PIXELS=512×28²
"""
import os

_VARIANT = os.environ.get("PIPELINE_VARIANT", "full_run")

# ── GPU 환경 ────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

# ── 모델 ────────────────────────────────────────────────────────────────────
MODEL_NAME           = os.environ.get("ACCIDENT_MODEL_NAME", "Qwen/Qwen3.5-9B")
VLLM_MAX_MODEL_LEN   = int(os.environ.get("ACCIDENT_VLLM_MAX_MODEL_LEN", "32768"))
VLLM_GPU_MEMORY_UTIL = float(os.environ.get("ACCIDENT_VLLM_GPU_MEMORY_UTIL", "0.90"))

# ── 경로 ────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT          = os.environ.get("ACCIDENT_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
RAW_DATA_ROOT      = os.environ.get("ACCIDENT_RAW_DATA_ROOT", os.path.join(DATA_ROOT, "raw"))
DOWNLOAD_ROOT      = os.environ.get("ACCIDENT_DOWNLOAD_ROOT", os.path.join(DATA_ROOT, "downloads"))
VIDEO_DIR          = os.environ.get("ACCIDENT_VIDEO_DIR", os.path.join(DATA_ROOT, "videos"))
VIDEO_MANIFEST_PATH = os.environ.get(
    "ACCIDENT_VIDEO_MANIFEST",
    os.path.join(DATA_ROOT, "video_manifest.csv"),
)
OUTPUT_ROOT        = os.environ.get("ACCIDENT_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "output"))

# ── Stage 1 — 사고 시각 초기 탐지 ──────────────────────────────────────────
STAGE1_TARGET_FPS  = 2.0
STAGE1_MAX_FRAMES  = 196 if _VARIANT == "full_run" else 120   # full_run=196, 그 외=120
STAGE1_MAX_RETRIES = 2
STAGE1_MIN_SEC     = 0.10

# ── Stage 2 — 사고 시각 정밀화 ─────────────────────────────────────────────
STAGE2_TARGET_FPS  = 6.0
STAGE2_MAX_FRAMES  = 120
STAGE2_MAX_RETRIES = 3
STAGE2_CLIP_SEC    = 4.0

# Stage 2 clip variant (±2초 클립 고해상도 분석)
STAGE2_CLIP_WINDOW_SEC = 2.0
STAGE2_CLIP_TARGET_FPS = 10.0
STAGE2_CLIP_MAX_FRAMES = 40
STAGE2_CLIP_MAX_PIXELS = 512 * 28 * 28   # 401,408 (633×633)

# ── Stage 3 — 사고 유형 분류 ────────────────────────────────────────────────
STAGE3_TARGET_FPS = 10.0 if _VARIANT == "tokens" else 6.0    # tokens=10fps, 그 외=6fps
STAGE3_MAX_FRAMES = 40   if _VARIANT == "tokens" else 32      # tokens=40프레임, 그 외=32

# ── vLLM 샘플링 ─────────────────────────────────────────────────────────────
MAX_NEW_TOKENS      = 320
MAX_NEW_TOKENS_TIME = 96
STAGE1_TEMPERATURE  = 0.1
TEMPERATURE         = 0.0
TOP_P               = 0.9
REPETITION_PENALTY  = 1.05
MAX_CALL_RETRIES    = 3

# ── 비전 (픽셀 예산) ────────────────────────────────────────────────────────
MIN_PIXELS            = 256  * 28 * 28   # 200,704 (448×448) — Stage 1 time flow 공통
MAX_PIXELS            = 512  * 28 * 28 if _VARIANT == "tokens" else 1024 * 28 * 28
#                       tokens=401,408 (633×633) / 그 외=802,816 (896×896) — Stage 3/4
TIME_STAGE_MAX_PIXELS = MIN_PIXELS       # Stage 1 time flow는 항상 448×448

# ── 디버그 ───────────────────────────────────────────────────────────────────
DEBUG_VISIBLE_EVIDENCE = False
MAX_EVIDENCE_CHARS     = 220

# ── 실행 옵션 ────────────────────────────────────────────────────────────────
SKIP_EXISTING = True

# ── 사고 유형 ────────────────────────────────────────────────────────────────
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}
