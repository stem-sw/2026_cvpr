# 재현 파이프라인 설정 요약

## 실행 구조

```
run_reproduce.py
  ├── [Step 1] run.py         → {run_id}_full/   (full 방식)
  ├── [Step 2] run_clip.py    → {run_id}_clip/   (clip 방식)
  └── [Step 3] min_ensemble   → {run_id}/submission.csv
```

---

## 설정값 비교

### Stage 1 — 사고 시각 초기 탐지

| 설정 | full_run | clip |
|------|----------|------|
| PIPELINE_VARIANT | `full_run` | `clip` |
| STAGE1_TARGET_FPS | 2.0 | 2.0 |
| STAGE1_MAX_FRAMES | **196** | **120** |
| STAGE1_MAX_RETRIES | 2 | 2 |
| STAGE1_MIN_SEC | 0.10 | 0.10 |
| STAGE1_TEMPERATURE | 0.1 | 0.1 |
| TIME_STAGE_MAX_PIXELS | 256×28²=200,704 (448×448) | 256×28²=200,704 (448×448) |

> full_run은 최대 196프레임 (긴 영상 커버), clip은 120프레임으로 토큰 절약

---

### Stage 2 — 사고 시각 정밀화

| 설정 | full_run | clip |
|------|----------|------|
| Stage2 방식 | 전체 영상 재샘플링 | **±2초 클립 고해상도 분석** |
| STAGE2_TARGET_FPS | 6.0 | 6.0 |
| STAGE2_MAX_FRAMES | 120 | 120 |
| STAGE2_MAX_RETRIES | 3 | 3 |
| STAGE2_CLIP_SEC | 4.0 | 4.0 |
| STAGE2_CLIP_WINDOW_SEC | - | **2.0** (±2초) |
| STAGE2_CLIP_TARGET_FPS | - | **10.0** |
| STAGE2_CLIP_MAX_FRAMES | - | **40** |
| STAGE2_CLIP_MAX_PIXELS | - | **512×28²=401,408** |
| TIME_STAGE_MAX_PIXELS | 256×28²=200,704 | 256×28²=200,704 |

> clip 방식은 Stage1 Top-3 후보 각각에 대해 ±2초 클립을 고해상도/고프레임으로 분석

---

### Stage 3 — 사고 유형 분류

| 설정 | full_run | clip |
|------|----------|------|
| STAGE3_TARGET_FPS | 6.0 | 6.0 |
| STAGE3_MAX_FRAMES | 32 | 32 |
| MAX_PIXELS | 1024×28²=802,816 (896×896) | 1024×28²=802,816 (896×896) |
| 입력 | accident_time (Stage2 결과) | accident_time (Stage2 결과) |

---

### Stage 4 — 충돌 위치 추론

| 설정 | full_run | clip |
|------|----------|------|
| MAX_PIXELS | 1024×28²=802,816 (896×896) | 1024×28²=802,816 (896×896) |
| 입력 | key_frame + Stage3 type 힌트 | key_frame + Stage3 type 힌트 |

---

### 공통 (vLLM)

| 설정 | 값 |
|------|----|
| MODEL_NAME | Qwen/Qwen3.5-9B |
| VLLM_MAX_MODEL_LEN | 32,768 |
| VLLM_GPU_MEMORY_UTIL | 0.90 |
| MAX_NEW_TOKENS | 320 |
| MAX_NEW_TOKENS_TIME | 96 |
| TEMPERATURE | 0.0 (Stage2/3/4) |
| STAGE1_TEMPERATURE | 0.1 |
| TOP_P | 0.9 |
| REPETITION_PENALTY | 1.05 |
| MAX_CALL_RETRIES | 3 |
| SKIP_EXISTING | True |

---

### min_ensemble

| 항목 | 내용 |
|------|------|
| 입력 | `{run_id}_full/submission.csv` + `{run_id}_clip/submission.csv` |
| accident_time | `min(full, clip)` |
| center_x, center_y | full_run 기준 유지 |
| type | full_run 기준 유지 |
| 출력 | `{run_id}/submission.csv` |

---

## 토큰 예산

| Stage | 방식 | 프레임 | 토큰/프레임 | 총 토큰 |
|-------|------|--------|------------|---------|
| Stage 1 | full | ≤196 | 256 (448×448) | ≤50,176 ⚠️ |
| Stage 1 | clip | ≤120 | 256 (448×448) | ≤30,720 ✅ |
| Stage 2 | full | ≤120 | 256 (448×448) | ≤30,720 ✅ |
| Stage 2 | clip | ≤40 | 512 (633×633) | ≤20,480 ✅ |
| Stage 3 | 공통 | ≤32 | 1024 (896×896) | ≤32,768 ✅ |
| Stage 4 | 공통 | 1 | 1024 (896×896) | 1,024 ✅ |

> full_run Stage1은 이론상 32K 초과 가능 (196×256=50,176). 실제로는 대부분 영상이 60초 이하라 120프레임 이내.

---

## 사용법

```bash
cd /workspace/sw/05_Accident_cvpr
conda activate qwen35

# 전체 실행
python src/run_reproduce.py --run-id reproduce_v1

# 테스트 (10개)
python src/run_reproduce.py --run-id reproduce_test --limit 10

# clip부터 재실행
python src/run_reproduce.py --run-id reproduce_v1 --from-step 2

# min_ensemble만
python src/run_reproduce.py --run-id reproduce_v1 --from-step 3
```
