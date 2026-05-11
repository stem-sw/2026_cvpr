# GitHub Publishing Checklist

## 목적

루트 공개 파이프라인을 기준으로, 다른 사용자가 데이터를 세팅하고 실행해서 결과를 재현할 수 있는 공개용 저장소 상태까지 정리한다.

---

## 현재 기준

- 공개 기준 파이프라인: 루트 `src/`
- 데이터 원본 입력: 저장소 루트 `accident.zip`
- 보고서/실험 기록: `98_Report/`, `99_Archive/`
- 최우선 목표:
  - 실행 경로 단순화
  - 하드코딩 경로 제거
  - 데이터 세팅 자동화
  - 중간 재현성 확인 가능 상태 만들기
  - README만 보고 실행 가능한 수준으로 문서화

---

## 0. 먼저 결정할 것

- [x] 데이터 파일명을 Kaggle 원본 그대로 유지하지 않고, 공개 스크립트에서 rename까지 책임지도록 확정
- [ ] 기존 인덱싱 방식이 꼭 필요한 코드가 남아 있는지 확인
- [x] 공개 레포에 포함하지 않을 항목 정리
  - 제외 확정: `99_Archive/`, `97_reproduce/output/`, 대용량 산출물, 개인 토큰, 로컬 전용 스크립트, 임시 실험 로그

완료 기준:

- 파일명 정책이 1개로 확정되어 이후 스크립트와 README가 그 기준만 사용함

---

## 1. 공개 레포 구조 확정

- [x] GitHub 루트에 남길 디렉터리/파일은 재현용 최소 구조로 정리하는 방향으로 확정
- [x] 루트 디렉터리를 공개용 메인 엔트리로 승격
- [x] `99_Archive`는 공개 대상에서 제외하기로 결정
- [x] 결과물 저장 위치를 `output/`로 일관화
- [x] 데이터 저장 위치를 루트 `data/` 단일 경로로 통일하기로 결정

권장 산출물:

- `README.md`
- `setup_data.sh`
- `requirements.txt`
- `src/`
- `.gitignore`
- `data/` 또는 `data/README.md`

완료 기준:

- 처음 보는 사람이 루트 구조만 보고 어디서 실행해야 하는지 알 수 있음

---

## 1.5. GitHub 레포 연결 확인

- [x] 공개 기준 remote를 `origin`으로 유지하기로 결정
- [x] `git remote -v`로 fetch/push URL이 의도한 GitHub 레포인지 확인
- [x] 기본 브랜치가 공개 기준 브랜치(`main` 등)와 맞는지 확인
- [ ] 현재 인증 상태에서 fetch 또는 push 권한 확인
  - `git fetch origin` 성공
  - `git push --dry-run origin main`은 로컬 Git credential helper가 삭제된 VS Code server node 경로를 참조해 실패
  - 실패 원인: `fatal: could not read Username for 'https://github.com': No such device or address`
  - 코드/remote 문제가 아니라 이 실행환경의 GitHub 인증 설정 문제로 판단
- [x] 기존 `backup` remote 제거 완료
- [x] 조직 레포(`stem-sw/2026_cvpr`)로 공개하기로 확정

현재 체크 포인트:

- 현재 remote는 `origin=https://github.com/stem-sw/2026_cvpr.git`
- `backup` remote는 제거되어 더 이상 다른 사용자 저장소로 작업이 나가지 않음
- 기본 브랜치는 현재 `main`
- `git fetch origin`은 성공
- push dry-run은 현재 컨테이너의 credential helper 문제로 실패했으므로, 공개 직전 GitHub 인증 설정을 다시 잡아야 함

권장 확인 명령:

- `git remote -v`
- `git branch --show-current`
- `git fetch origin`
- `git push --dry-run origin main`

완료 기준:

- 공개 대상 레포, 기본 브랜치, 권한 상태가 모두 확정되어 마지막 업로드 단계에서 막히지 않음

---

## 2. 데이터 세팅 자동화

- [x] 루트 `setup_data.sh` 검토 및 보완
- [x] 사용자가 준비한 zip 기준으로 압축 해제부터 인덱싱까지 자동화
- [x] 최종 비디오 위치가 파이프라인 입력 경로와 정확히 맞도록 정리
- [x] 파일명 정책에 따라 rename 단계 포함 여부 확정
- [x] 실패 시 필요한 선행조건 출력
  - 저장소 루트의 `accident.zip` 또는 명시적 zip 경로
  - `unzip`
  - 디스크 용량

체크 포인트:

- [x] 저장소 루트의 `accident.zip` 기준으로 `setup_data.sh`가 비디오 폴더를 준비하는지 확인
- [x] 루트 `src` 실행 코드가 그 경로를 그대로 읽는지 확인

완료 기준:

- 문서 없이도 `setup_data.sh` 실행 후 파이프라인 입력 데이터가 준비됨

---

## 3. 경로 리팩토링

- [x] 코드 내 절대 경로(`/root/Desktop/workspace/...`) 전수 검색
- [x] 상대 경로 또는 환경변수 기반 경로로 변경
- [x] `src/config.py`에 경로 설정을 일원화
- [x] `VIDEO_DIR`, `OUTPUT_ROOT`를 로컬 환경에 덜 의존하게 정리
- [x] `HF_TOKEN` 하드코딩 제거

우선 확인 파일:

- `src/config.py`
- `src/run_reproduce.py`
- `src/run.py`
- `src/run_clip.py`
- `src/utils/io_utils.py`

체크 포인트:

- [ ] 프로젝트 루트를 옮겨도 실행 경로가 깨지지 않는지 확인
- [x] CLI 인자 없이 기본 실행했을 때도 합리적인 기본 경로를 참조하는지 확인

완료 기준:

- 개인 PC 경로, 개인 토큰 없이 실행 가능

---

## 4. 중간 재현성 확인 체계 만들기

목표:

- 전체 20시간짜리 실행 전에, 작은 샘플로 단계별 결과가 유지되는지 빠르게 확인할 수 있어야 함

체크리스트:

- [x] 고정 smoke test 규칙 정하기
  - 기준: `--limit 10`
- [x] 기준 run 하나 생성
  - `python src/run_reproduce.py --run-id smoke_baseline --limit 10`
  - 완료 기준 run: `CUDA_VISIBLE_DEVICES=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True conda run -n qwen35 python src/run_reproduce.py --run-id smoke_baseline_qwen35 --limit 10`
- [x] 짧은 runtime 검증 run 수행
  - `CUDA_VISIBLE_DEVICES=1 conda run -n qwen35 python src/run_reproduce.py --run-id smoke_one_gpu1 --limit 1`
  - 확인 결과: `output/smoke_one_gpu1_full/stage1.csv` 생성, Stage 2 진입 확인
- [x] 아래 파일을 기준 산출물로 보관
  - `output/smoke_baseline_qwen35_full/stage2.csv`
  - `output/smoke_baseline_qwen35_clip/stage2.csv`
  - `output/smoke_baseline_qwen35/stage2.csv`
  - `output/smoke_baseline_qwen35/stage3.csv`
  - `output/smoke_baseline_qwen35/stage4.csv`
  - `output/smoke_baseline_qwen35/submission.csv`
  - 현재 모두 `11` lines(header 포함)로 생성 확인
- [x] 수정 후 같은 조건으로 재실행
  - 완료 기준 run: `CUDA_VISIBLE_DEVICES=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True ACCIDENT_VLLM_GPU_MEMORY_UTIL=0.85 conda run -n qwen35 python src/run_reproduce.py --run-id smoke_cleanup_qwen35 --limit 10`
  - 최종 상태: `status=0`, 총 소요시간 `0시간 24분 35초`
- [x] 단계별 diff 확인
  - 행 수
  - `path`
  - `accident_time`
  - `type`
  - `center_x`, `center_y`
  - 확인 결과: baseline 대비 주요 CSV 6개 모두 `rows 10->10`, `columns_equal=True`, `ids_equal=True`

권장 검증 순서:

1. `setup_data.sh` 수정 후 `--limit 10`
2. 경로 리팩토링 후 `--limit 10`
3. 코드 cleanup 후 `--limit 10`
4. README/requirements 정리 전 `--limit 100`
5. publish 직전 full run 1회

주의:

- Stage 1이 샘플링 영향을 받으면 완전 동일 수치가 안 나올 수 있음
- 그 경우 먼저 구조 재현성부터 확인
  - 파일 생성 여부
  - 행 수
  - 단계 진행 성공 여부

완료 기준:

- 코드 수정 후 어디서 결과가 달라졌는지 stage 단위로 바로 알 수 있음

---

## 5. 코드 정리

- [x] 사용하지 않는 import 제거
- [x] 명확한 죽은 코드 제거
  - 공개 파이프라인에서 호출되지 않는 LoRA 로딩 분기 제거
  - 공개용 `src/`에서 호출되지 않는 작은 경로/처리 헬퍼 제거
- [x] 로그 정리
  - vLLM 응답 원문 일부를 매번 출력하던 로그를 핵심 필드 요약으로 변경
- [ ] 주석 처리된 대형 블록 삭제
- [ ] 중복 로직이 있으면 최소 범위에서 정리

체크 포인트:

- [x] 공개용 `src/` 기준 명확한 unused import 및 미사용 헬퍼 정리
- [x] cleanup 후 `compileall`, `run_reproduce.py --help`, unused import 재점검 통과
- [x] cleanup 전후 smoke test 결과가 의도치 않게 변하지 않았는지 확인

완료 기준:

- 공개 저장소에서 읽는 사람이 핵심 흐름을 따라갈 수 있음

---

## 6. 실행 흐름과 문서 일치시키기

- [x] `README.md` 내용이 실제 코드와 맞는 방향으로 정리
- [x] `run_reproduce.py` 기준 실행 흐름으로 문서화
- [x] 수동 단계가 자동화되었으면 문서에서 제거
- [x] GPU 요구사항과 권장 실행 명령 정리
- [x] 예상 실행 시간, 디스크 사용량 정리
- [x] 오류 발생 시 확인할 항목 정리

문서에 꼭 들어갈 내용:

- 프로젝트 소개
- 폴더 구조
- 환경 준비
- `accident.zip` 배치 위치와 데이터 준비 방법
- 단일 명령 재현 방법
- 단계별 재개 방법
- 산출물 위치
- 재현성 확인 방법

체크 포인트:

- [x] README만 읽고 새 사용자가 실행 순서를 따라갈 수 있는지 점검

완료 기준:

- 구두 설명 없이도 실행 가능

---

## 7. 환경 파일 정리

- [x] 실제 필요한 패키지만 추려서 `requirements.txt` 생성
- [x] 버전 고정 필요 패키지 정리
- [x] GPU/추론 엔진 의존성 명시
- [ ] 선택 의존성과 필수 의존성 구분
  - 현재 공개 실행 기준 의존성은 `requirements.txt`에 고정
  - 더 세밀한 optional 분리는 후속 정리 항목으로 남김

체크 포인트:

- [x] 새 가상환경에서 설치 명령이 동작하는지 확인
- [x] 설치 후 import 에러 없이 smoke test가 도는지 확인

완료 기준:

- 다른 사용자가 환경 설치에서 막히지 않음

---

## 8. 공개 전 최종 점검

- [x] `.gitignore` 정리
- [x] 대용량 output 파일 제외
- [x] API 키, 토큰, 개인 경로 제거
- [x] 불필요한 캐시 파일 제거
  - `__pycache__`
  - 임시 로그
  - 중간 산출물
- [ ] 라이선스 파일 추가 여부 결정
- [ ] 공개 설명 문구와 프로젝트 이름 확정

최종 확인:

- [ ] clean clone 후 실행 절차가 성립하는지 확인
- [x] smoke test 통과
- [ ] 중간 규모 테스트 통과
  - 이번 진행에서는 사용자 요청에 따라 `--limit 100` 검증은 생략
- [ ] full run 또는 대표 산출물 비교 완료

완료 기준:

- 외부 사용자가 clone 후 문서대로 재현 가능

---

## 추천 실제 진행 순서

1. 파일명 정책 결정
2. `setup_data.sh` 정리
3. 경로 리팩토링
4. smoke baseline 생성
5. 코드 cleanup
6. README와 실행 흐름 동기화
7. requirements 정리
8. 최종 재현성 점검
9. 공개 준비

---

## 진행 기록

- [x] 2026-05-06 체크리스트 초안 작성
- [x] 파일명 정책 확정
- [x] 데이터 세팅 자동화 완료
- [x] 경로 리팩토링 완료
- [x] smoke baseline 확보
- [x] cleanup 후 smoke 재검증 완료
- [x] README 정리 완료
- [x] requirements 정리 완료
- [ ] publish 전 최종 검증 완료
  - `--limit 10` cleanup smoke 검증은 완료
  - 사용자 요청에 따라 `--limit 100` 검증은 생략
  - push dry-run은 credential helper 문제로 미완료
