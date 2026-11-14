# Dual camera pairwise refinement

현재 두 영상의 RoMa RGB 대응, 동결 로봇 모델의 voting 관측과 카메라별 관절각을 결합하여 로봇 포즈·상대 카메라 포즈·배경 3D점을 공동 추정하는 Robotpose 확장입니다.

구현, 방법론, 전체 실험 결과와 지표 정의는 **[현재 버전 통합 문서](docs/CURRENT_IMPLEMENTATION_REPORT.html)** 한 문서에 정리되어 있습니다. HTML 파일을 내려받아 브라우저에서 열면 됩니다. 집계 원자료는 `results/`에 있습니다.

## 실행 전 준비

이 폴더는 독립적인 로봇 모델 구현이 아닙니다. 원래 Robotpose의 다음 소스를 그대로 읽어야 합니다.

- `robotpose/network3.py`, `robotpose/vote2.py`
- `robotpose/my_pnp.py`, `robotpose/infer_live.py`
- `ur5e_capture/geometry.py`

이 확장을 Robotpose 바로 아래에 배치하거나 `ROBOTPOSE_REPO_ROOT`로 원본 저장소를 지정합니다. 원본 소스가 없으면 명확한 오류를 내고 중단합니다. 데이터셋, 로봇·RoMa 가중치, 제3자 소스, 실행 캐시는 배포에 포함하지 않습니다.

RoMaV2는 DINOv3 특징 백본을 사용합니다. 고정 소스의 이용 조건과 출처는 [RoMaV2 MIT License](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/LICENSE)와 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/LICENSE.md)에서 확인할 수 있습니다.

검증 환경은 Python 3.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.6, SciPy 1.15.3, OpenCV 4.13.0.92, PyYAML 6.0.3, matplotlib 3.10.9, Pillow 12.2.0, pyrr 0.10.3입니다. 원래 Robotpose의 해당 의존성이 설치된 Python을 사용합니다. 원래 환경을 자동 수정하지 않습니다. 실제 로봇 forward와 RoMa matching에는 CUDA가 필요합니다.

다음 명령은 현재 작업 디렉터리에 `dual_camera_pairwise_v2/`가 있는 배치를 가정합니다. 아래 두 경로만 실제 설치 위치에 맞게 지정합니다.

```bash
export ROBOTPOSE_REPO_ROOT=/path/to/Robotpose
DCP_PY="$ROBOTPOSE_REPO_ROOT/.venv/bin/python"
DCP=dual_camera_pairwise_v2

# CPU·합성 검증
"$DCP_PY" -B "$DCP/run.py" test

# 공식 RoMa/DINOv3 소스와 고정 가중치·작은 의존성을 확장 안에 준비
# 첫 실행에는 네트워크 연결이 필요합니다.
"$DCP_PY" -B "$DCP/setup_matchers.py"
```

## UR_dual_2596 전체 재현

현재 fresh importer는 UR recorder 형식의 **2,596쌍** 및 지정 checkpoint의 SHA-256을 엄격히 검사합니다. checkpoint는 원래 `trained_models/ur5e_new2/last.pt`이며 SHA-256은 `7b5e518c15a54a7e5146d8b658a2d71f3e185866b8be69f9e8409df2ccc10d8c`입니다. 경로가 다르면 `--dataset`, `--checkpoint`를 지정할 수 있지만 내용 검사 기준은 같습니다.

모든 출력은 확장 내부에 생성됩니다. 이미 완성된 결과 디렉터리는 덮어쓰지 않으므로 반복 실험은 새로운 출력 경로를 사용합니다. importer의 동일 계약 재개만 허용됩니다.

```bash
# 동결 로봇 모델 CUDA FP32 추론 및 원본 voting 후보 생성
"$DCP_PY" -B "$DCP/ingest/ingest.py" \
  --config "$DCP/configs/ur_dual_2596_pairwise.yaml" \
  --output "$DCP/ingest/full"

# 평가 reference는 별도 명령으로 준비
"$DCP_PY" -B "$DCP/ingest/ingest.py" \
  --prepare-reference --output "$DCP/ingest/full"

# 현재 영상쌍의 RoMa 대응
"$DCP_PY" -B "$DCP/matcher_worker.py" \
  --pairs "$DCP/ingest/full/pairs.jsonl" \
  --observations "$DCP/ingest/full/observations.json" \
  --config "$DCP/configs/ur_dual_2596_pairwise.yaml" \
  --backend roma_v2 --split all \
  --output "$DCP/cache/matches/roma_v2/manifest.json"

# 동일 관측으로 다섯 방법 모두 추론; workers는 CPU 자원에 맞게 조정
"$DCP_PY" -B "$DCP/run.py" refine-pairs \
  --pairs "$DCP/ingest/full/pairs.jsonl" \
  --observations "$DCP/ingest/full/observations.json" \
  --config "$DCP/configs/ur_dual_2596_pairwise.yaml" \
  --matches "$DCP/cache/matches/roma_v2/manifest.json" \
  --methods all --split all --workers 12 \
  --output "$DCP/runs/roma_v2_full2596"

# 포즈별 측정치를 만든 후 최종 전체 집계
"$DCP_PY" -B "$DCP/run.py" evaluate-pairs \
  --pairs "$DCP/ingest/full/pairs.jsonl" \
  --reference-manifest "$DCP/ingest/full/reference.jsonl" \
  --run "$DCP/runs/roma_v2_full2596" --split all \
  --output "$DCP/reports/evaluation_roma_v2_all"
"$DCP_PY" -B "$DCP/evaluate_all_success.py" \
  --pairs "$DCP/ingest/full/pairs.jsonl" --dataset UR_dual_2596 \
  --source "roma_v2=$DCP/reports/evaluation_roma_v2_all" \
  --output "$DCP/reports/metrics_roma_v2_all"

# 기존 infer_dir와 호환되는 열 순서의 stage CSV
"$DCP_PY" -B "$DCP/export_infer_csv.py" \
  --pairs "$DCP/ingest/full/pairs.jsonl" \
  --observations "$DCP/ingest/full/observations.json" \
  --run "$DCP/runs/roma_v2_full2596" \
  --output "$DCP/reports/infer_csv_roma_v2"
```

최종 수치 파일은 `reports/metrics_roma_v2_all/summary_metrics.csv`, `roma_v2_all/paired_comparisons.csv`, `roma_v2_all/curves.json`입니다. 내보낸 CSV와 상태표의 경로는 `reports/infer_csv_roma_v2/manifest.json`에 기록됩니다. `dual_no_cross`는 RGB 항을 제거한 대조군입니다.

## 재현 범위

UR_dual_2596은 위 명령으로 원본 이미지에서부터 전체 과정을 실행할 수 있습니다. UR_dual_889의 전체 집계도 문서와 `results/`에 포함되어 있습니다. 889 실측의 동일 입력 재현에는 해당 실험에 사용된 이미지, raw voting 관측·후보·pair manifest가 별도로 필요합니다. 이 배포에는 그 캐시를 포함하지 않으며, 현재 2596 전용 importer를 889용으로 일반화하지 않았습니다.

`pairwise/solver.py`, `geometry.py`, `candidates.py`, `matching.py`, `evaluation.py`와 CSV 내보내기의 수치 코드는 실측 버전과 동일합니다. 배포 사본에는 경로 이동, 원본 저장소 경로 지정, 전체 데이터 전용 CLI, 기본 RoMa 준비만 적용했습니다. 검증 근거는 `validation/`의 작은 JSON 파일에 기록됩니다.
