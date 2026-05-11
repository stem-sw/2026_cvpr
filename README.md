# 05_Accident_cvpr

이 저장소는 공개용 재현 진입점입니다.  
사용자가 준비한 `accident.zip`을 저장소 루트에 두면 `data/videos/` 입력 구조로 자동 정리하고, `src/run_reproduce.py`로 full + clip + min-ensemble + Stage 3/4 파이프라인을 재현하는 것을 목표로 합니다.

## 디렉터리 구조

```text
05_Accident_cvpr/
├── accident.zip           # 로컬 입력 파일, Git 제외
├── data/
│   ├── raw/               # 압축 해제 원본
│   ├── videos/            # 파이프라인 입력용 indexed 비디오
│   └── video_manifest.csv # indexed_name ↔ original_name 매핑
├── output/                # 실행 결과
├── requirements.txt
├── setup_data.sh
└── src/
```

## 검증한 실행 환경

이 저장소는 아래 환경에서 실제 추론까지 확인했습니다.

- Conda env: `qwen35`
- Python: `3.10.20`
- GPU: RTX 3090 24GB x2
- `torch==2.10.0`
- `vllm==0.18.0`
- `transformers @ git+https://github.com/huggingface/transformers.git@c38b2fb78eaedd4261a0e446f7976345cd1c7f1b`

베이스 환경의 Python 3.8 + `transformers 4.46.x` 조합은 `Qwen/Qwen3.5-9B`를 인식하지 못해 재현 기준 환경으로 사용하지 않습니다.

## 환경 준비

권장 설치:

```bash
conda activate qwen35
pip install -r requirements.txt
```

또는 실행만 할 때는:

```bash
conda run -n qwen35 python src/run_reproduce.py --help
```

모델 접근에 토큰이 필요한 환경이면 실행 전에 설정합니다.

```bash
export HF_TOKEN=your_token_here
```

GPU 0이 이미 사용 중이면 GPU를 바꿔서 실행합니다.

```bash
export CUDA_VISIBLE_DEVICES=1
```

vLLM 메모리 비율을 조정하고 싶으면:

```bash
export ACCIDENT_VLLM_GPU_MEMORY_UTIL=0.85
```

모델 호스팅 연결 확인을 건너뛰고 싶으면:

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

## 데이터 준비

`setup_data.sh`는 아래 작업을 자동으로 수행합니다.

1. 저장소 루트의 `accident.zip` 또는 전달한 zip 경로 찾기
2. `data/raw/` 아래 압축 해제
3. 원본 비디오 정렬
4. `0000_original.mp4` 형식으로 `data/videos/` 준비
5. `video_manifest.csv` 생성

권장 방식:

```bash
cd /path/to/05_Accident_cvpr
bash setup_data.sh
```

직접 경로를 넘겨도 됩니다.

```bash
bash setup_data.sh /absolute/path/to/accident.zip
```

검증한 로컬 데이터 기준:

- 입력 zip: 약 16GB
- 압축 해제 및 인덱싱 후 `data/`: 약 19GB
- 비디오 수: 2027개

## 기본 실행

메인 재현 진입점:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_reproduce.py --run-id reproduce_v1
```

이 스크립트는 아래 순서로 실행됩니다.

1. full variant Stage 1 time flow
2. clip variant Stage 1 time flow
3. min-ensemble
4. Stage 3/4

Stage 1 time flow는 `stage1.csv`를 만든 뒤 같은 흐름에서 정밀화 결과인 `stage2.csv`까지 생성합니다. 이후 Stage 3/4는 기존처럼 `stage2.csv`를 입력으로 사용합니다.

최종 결과:

```text
output/reproduce_v1/submission.csv
```

## Smoke Test

전체 실행 전에는 작은 샘플부터 확인하는 것을 권장합니다.

```bash
CUDA_VISIBLE_DEVICES=1 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
conda run -n qwen35 python src/run_reproduce.py --run-id smoke_baseline_qwen35 --limit 10
```

이 세션에서 확인한 상태:

- `bash setup_data.sh` 성공
- `data/videos/` 2027개 인덱싱 완료
- `output/smoke_one_gpu1_full/stage1.csv` 생성 확인
- `smoke_one_gpu1` 기준으로 Stage 1 time flow 진입 확인
- `smoke_baseline_qwen35` 기준으로 full + clip + min-ensemble + Stage 3/4 전체 완주 확인
- `output/smoke_baseline_qwen35/submission.csv` 생성 확인

즉 현재 기준으로는 10개 샘플 smoke test가 끝까지 완료됐고, 공개용 재현 구조가 실제로 동작함을 확인했습니다.

최근 cleanup 이후에도 같은 10개 샘플 기준으로 다시 검증했습니다.

- run id: `smoke_cleanup_qwen35`
- 최종 상태: `status=0`
- 소요 시간: 약 24분 35초
- 주요 CSV 6개 모두 header 포함 11줄 생성
- cleanup 전 baseline 대비 행 수, 컬럼, 입력 id 구조 일치 확인

전체 실행 시간은 GPU/네트워크/모델 캐시 상태에 따라 달라집니다. 이 저장소에서는 먼저 `--limit 10`으로 환경과 데이터 경로를 검증한 뒤 full run을 시작하는 방식을 권장합니다.

## 재개 실행

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_reproduce.py --run-id reproduce_v1 --from-step 2
CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_reproduce.py --run-id reproduce_v1 --from-step 3
CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_reproduce.py --run-id reproduce_v1 --from-step 4
```

개별 variant를 직접 실행할 때 `--only-stage 1`은 시간 추론 전체 흐름을 의미하며, `stage1.csv`와 `stage2.csv`를 함께 준비합니다. 기존 호환을 위해 `--only-stage 2`는 `stage1.csv`가 이미 있을 때 정밀화만 다시 실행하는 내부용 경로로 남겨두었습니다.

## 개별 실행 진입점

- `CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run.py --run-id full_only`
- `CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_clip.py --run-id clip_only`
- `CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_full_ensemble.py --run-id full_ens_v1`
- `CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_tokens.py --run-id tokens_v1`

모든 진입점은 기본적으로 저장소 루트의 `data/videos/`를 입력으로 사용합니다.

## 경로 환경변수

기본 경로를 바꾸고 싶을 때만 사용합니다.

```bash
export ACCIDENT_DATA_ROOT=/path/to/data
export ACCIDENT_VIDEO_DIR=/path/to/data/videos
export ACCIDENT_OUTPUT_ROOT=/path/to/output
```

기본값:

- `ACCIDENT_DATA_ROOT` → `data`
- `ACCIDENT_VIDEO_DIR` → `data/videos`
- `ACCIDENT_OUTPUT_ROOT` → `output`

## 주의

- `accident.zip`, `data/`, `output/`은 공개 저장소에 포함하지 않는 것을 권장합니다.
- 현재 파이프라인은 Qwen/vLLM GPU 추론을 전제로 합니다.
- GPU 0이 이미 사용 중이면 `CUDA_VISIBLE_DEVICES=1`처럼 명시적으로 비어 있는 GPU를 선택하세요.

## 문제 해결

- `setup_data.sh: No such file or directory`가 나오면 저장소 루트가 아니라 다른 디렉터리에서 실행한 것입니다. `cd /path/to/05_Accident_cvpr` 후 다시 실행하세요.
- `accident.zip not found`가 나오면 저장소 루트에 `accident.zip`을 두거나 `bash setup_data.sh /absolute/path/to/accident.zip`처럼 zip 경로를 직접 넘기세요.
- Kaggle API 인증 오류가 나도 이 공개 재현 흐름에서는 Kaggle CLI가 필요하지 않습니다. 사용자가 직접 준비한 `accident.zip`에서 압축 해제와 인덱싱을 진행합니다.
- Qwen 모델을 찾지 못하거나 `transformers` 관련 오류가 나면 README의 검증 환경과 `requirements.txt` 기준으로 환경을 맞추세요.
- vLLM 메모리 부족이 나면 `CUDA_VISIBLE_DEVICES`로 빈 GPU를 고르거나 `ACCIDENT_VLLM_GPU_MEMORY_UTIL=0.80`처럼 메모리 사용률을 낮춰서 다시 실행하세요.
- `Cannot send a request, as the client has been closed` 같은 일시적 vLLM client 로그가 나올 수 있습니다. 재시도 후 최종 `status=0`으로 끝나면 실패로 보지 않습니다.
