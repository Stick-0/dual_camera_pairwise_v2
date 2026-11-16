# Candidates 배경 결합 × Essential 초기화 통제 실험

현재 구현의 R0/R1/E0/E1 통제 실험 코드와 **UR_dual_2596 전체 2,596쌍 / 5,192개 카메라 관측**의 실제 결과입니다.

- [최종 상세 HTML 보고서](reports/final/ablation_report.html): 내려받아 브라우저에서 열면 실제 로봇 사진 11개와 선택형 결과표를 볼 수 있습니다.
- [집계 결과](reports/final/summary_metrics.csv), [paired 비교](reports/final/paired_comparisons.csv), [전체 관측별 평가](reports/final/per_observation_metrics.csv)
- [실험 명세](CODEX_CANDIDATES_BACKGROUND_ABLATION.md), [배포 파일과 원본/배포 SHA](publication_manifest.json)

| 조건 | 초기화 | 배경 최적화 | 유효 관측/5,192 | ADD50 | AUC | 유효 포즈 평균 ADD |
|---|---|---|---:|---:|---:|---:|
| R0 | 로봇 단안 초기해 | OFF | 5,192 | 64.73% | 0.5643 | 46.39 mm |
| R1 | R0와 같은 A0/G0 | ON | 5,192 | 89.56% | 0.7581 | 24.25 mm |
| E0 | Essential 초기해 | OFF | 4,990 | 62.06% | 0.5408 | 47.36 mm |
| E1 | E0와 같은 A0/G0 | ON | 4,926 | 84.55% | 0.7157 | 24.93 mm |

ADD50/AUC는 실패를 포함하는 전체 분모이며 평균은 유효 포즈 조건부입니다. 각 조건은 공동 최적화 시작점 하나를 사용하고, R/E의 초기 포즈 배열과 책임도는 각각 OFF/ON 사이에서 공유합니다. OFF는 A/G의 12개 변수를 자유롭게 최적화하며 배경점 변수·배경 잔차·배경 gate가 없습니다. E 실패는 R 초기값으로 대체하지 않습니다. 기존 다중 시작점 production 결과는 별도 참고표로 유지합니다.

R1−R0 paired ADD는 −22.14 mm, 95% CI는 [−23.97, −20.37] mm입니다. E0−R0는 +1.37 mm이며, 상호작용 신뢰구간은 0을 포함합니다. 배경 gate에 의한 추가 탈락은 0개였습니다. 시간 블록 bootstrap은 원래 시간축의 pair를 묶어 seed 0, 2,000회, 1초 기본 블록 및 0.5/2초 민감도를 사용합니다. 전체 표와 한계는 상세 보고서에 있습니다.

## 배포 범위와 원본 보존

실험의 `run.py`와 `src/*.py`는 실제 실행 파일과 **바이트 단위로 동일**합니다. 수식·solver·평가 지표·CSV 수치는 배포 과정에서 변경하지 않았습니다. 원본 가중치·모델·RoMa·관측 후보를 유지하며 재학습·새 후보 생성은 수행하지 않았습니다. 새 확장 안에만 출력하는 runtime guard가 있습니다.

데이터셋, 가중치, raw 관측/매칭 캐시, pair별 전체 포즈/책임도 결과, 과거 출력, 대형 시간 trace, bootstrap NPZ는 포함하지 않습니다. 최종 HTML은 그림을 자체 포함하므로 중복 그림 파일도 제외했습니다. 개별 관측 ADD와 실패 집계는 포함합니다. 배포에서 제외된 보고서 링크는 “로컬 원본 보관 · 배포 제외”로 표시됩니다.

검증 JSON의 큰 파일별 해시 목록은 원본 개수와 집합 SHA로 요약했습니다. `${ROBOTPOSE_ROOT}`는 원래 로컬 경로를 대신하는 표시입니다. JSON 내부의 기존 source/result hashes는 **원본 실행 산출물**의 해시이며, 배포 파일 자체의 해시는 `publication_manifest.json`에 따로 있습니다. 보존 검사 통과 기록은 이 실험 완료 당시의 기록이고, clone 후 새 환경의 검증을 의미하지 않습니다.

## 실행에 필요한 원본 환경

이 저장소 clone만으로 실데이터 추론을 실행할 수 없습니다. 원본 Robotpose 소스, 해당 실행에서 검증한 UR_dual_2596 관측·RoMa 캐시·평가 기준·production 결과가 별도로 필요합니다. 원본 파일 안의 절대 입력 경로와 해시 계약도 유효해야 합니다. 데이터가 다른 위치로 이동했다면 캐시 manifest를 임의로 편집해서 해시 검사를 우회하지 않습니다.

동결 runner는 다음 배치를 요구합니다. 기존 `dual_camera_pairwise_v2/`를 GitHub의 경량 배포 폴더로 덮어쓰지 마세요. 원래 실행의 `pairwise/preservation.py`, 기존 실험 디렉터리와 보고서 등을 포함한 기존 소스를 읽어야 합니다.

```text
/path/to/Robotpose/
  .venv/
  ur5e_capture/geometry.py
  dual_camera_pairwise_v2/              # 원래 실행 소스와 입력 캐시, 읽기 전용
    experiments/ur_dual_2596_20261114/   # ingest/full, matches, runs, reports
  candidate_background_ablation_20261116/  # 이 새 확장만 배치
```

원본 root에 같은 ablation 폴더가 이미 있으면 복사하거나 덮어쓰지 않습니다. 그 환경에서는 이미 보존된 로컬 구현을 사용하세요. 다른 준비된 원본 환경에 배치할 때는 아래 명령처럼 목적지가 없음을 먼저 검사합니다.

```bash
export ROBOTPOSE_REPO_ROOT=/path/to/Robotpose
test ! -e "$ROBOTPOSE_REPO_ROOT/candidate_background_ablation_20261116" && \
  cp -R candidate_background_ablation_20261116 "$ROBOTPOSE_REPO_ROOT/"
cd "$ROBOTPOSE_REPO_ROOT"
DCP_PY="$ROBOTPOSE_REPO_ROOT/.venv/bin/python"
ABL=candidate_background_ablation_20261116

# 경로만 실제 절대경로로 치환한 새 local 설정 생성. 기존 파일이 있으면 실패합니다.
"$DCP_PY" -B "$ABL/configure_local.py" --robotpose-root "$ROBOTPOSE_REPO_ROOT"

# 데이터와 무관한 평가 단위검사
OPENBLAS_NUM_THREADS=1 "$DCP_PY" -B "$ABL/src/test_eval.py"

# 입력 캐시를 읽는 실제 사전 지정 4쌍 및 전체 2,596쌍 실행
"$DCP_PY" -B "$ABL/run.py" smoke --config "$ABL/configs/experiment.local.yaml" \
  --workers 4 --output "$ABL/results/smoke_reproduction"
"$DCP_PY" -B "$ABL/run.py" full --config "$ABL/configs/experiment.local.yaml" \
  --workers 12 --output "$ABL/results/full_reproduction"
"$DCP_PY" -B "$ABL/run.py" evaluate --config "$ABL/configs/experiment.local.yaml" \
  --run-dir "$ABL/results/full_reproduction" --output "$ABL/reports/reproduction"
```

모든 재실행 출력은 아직 없는 이름이어야 합니다. 기본 `configs/experiment.yaml`은 배포용 경로 템플릿입니다. 이를 runner에 직접 전달하지 말고 `configure_local.py`로 만든 절대경로 설정을 `--config`로 명시하세요. 이 과정은 source/inputs/reference의 경로 문자열만 치환하며 수치 설정은 유지합니다.

`src/validate_inputs.py`, `validate_controls.py`, `verify_full.py`, `finalize.py` 및 보존 스크립트는 원래 로컬 실행의 고정 경로와 출력 계약을 감사한 원본입니다. 일부는 기본 설정 또는 원래 `results/full`을 고정 참조하므로 새 배포 경로에 자동 적응하는 도구로 표시하지 않습니다. 원래 검증 결과는 `validation/`, `preservation/`에서 확인할 수 있습니다. 추정·평가 재현의 지원 진입점은 위 `run.py --config` 명령입니다.

실제 실행 환경은 Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, OpenCV 4.13.0, PyTorch 2.11.0+cu128이며 YAML/Matplotlib/Pillow 의존성도 사용합니다. 기존 Python 환경을 사용하며 자동 설치/변경은 하지 않습니다. 2026-11-16 실제 배치는 GPU 드라이버를 사용할 수 없어 검증된 기존 CUDA 관측·매칭 캐시를 재사용하여 CPU SciPy 12 workers로 약 187.494초 걸렸습니다. 이 시간은 모델 forward/매칭 시간을 포함하지 않습니다. `matching.device: cuda`는 기존 캐시 생성 조건입니다.

고정 설정은 후보 σ=2px, outlier ε=0.05, EM 최대 10회, 내부 solve마다 `max_nfev=60`, 배경 최대 80개·σ=1.5px·2D Huber δ=3·λ=1입니다. 기존 443,394개 파일의 내용 해시 변화가 없었던 원본 보존 검사 결과도 함께 제공합니다.
