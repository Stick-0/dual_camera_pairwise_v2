배포 안내: 전체 개별 이미지는 Release ZIP에 있습니다. https://github.com/Stick-0/dual_camera_pairwise_v2/releases/tag/visualization-20261120
Git 경량 사본만으로는 개별 이미지가 없습니다. README.md를 먼저 확인하세요.
아래 실행/검증 설명은 원래 로컬 생성 기록이며 배포본 자체 검사 결과가 아닙니다.

R0 / R1 전체 프레임 키포인트·포즈 좌표계 및 ADD 시각화
작성일: 2026-11-20

보기
  index.html을 브라우저에서 열면 프레임을 이동하면서 R0/R1과 두 카메라를 비교할 수 있습니다.
  인터넷이나 서버 없이 동작합니다. HTML과 이미지 폴더의 상대 경로를 유지하세요.

대상과 범위
  UR_dual_2596 전체 2,596쌍, 두 카메라, R0/R1 두 조건입니다.
  R0 = PnP 기반 로봇 포즈 초기화 + 배경 OFF.
  R1 = R0와 정확히 같은 초기 포즈 + 배경 ON.
  PnP 기반 초기화는 PnP 후 단안 Candidates EM 정제까지 포함합니다.
  각 조건의 본 공동 최적화 시작점은 1개이며, 기존 다중 시작점 production 결과는 섞지 않았습니다.
  원본 2026-11-16 통제 실험의 최종 저장 포즈와 관측을 읽어 시각화했습니다.
  모델 추론, 후보 생성, 매칭, PnP, 최적화, 재학습을 다시 수행하지 않았습니다.

개별 이미지: 총 10,384개 JPEG
  overlays/R0/camera1/000000.jpg ... 002595.jpg
  overlays/R0/camera2/000000.jpg ... 002595.jpg
  overlays/R1/camera1/000000.jpg ... 002595.jpg
  overlays/R1/camera2/000000.jpg ... 002595.jpg
  각 영상의 원래 640x480 화소 좌표를 유지하고 상단 52px, 하단 72px 범례를 추가했습니다.
  최종 저장 이미지 크기는 640x604입니다.
  overlays/manifest.jsonl은 모든 이미지의 원본 RGB SHA, 출력 SHA, 조건, 카메라, pair, ADD를 기록합니다.

기호와 좌표계
  주황 +: 원래 voting 입력점. raw valid mask를 그대로 적용했습니다.
  노란 빈 원: 최종 책임도에 따른 사후 MAP 후보. MAP valid mask를 그대로 적용했습니다.
  청록 점/선, 0~6 번호: 최종 추정 포즈로 투영한 FK 키포인트와 직렬 연결 골격입니다.
  FK 골격은 실제 링크 외곽이나 메쉬를 표시한 것이 아닙니다.
  빨강 X / 초록 Y / 파랑 Z: 추정된 로봇 베이스 좌표계. 축 길이는 모두 0.12m입니다.
  저장된 base-to-camera T로 원점 및 X/Y/Z 끝점을 변환하고 실제 카메라 K/D로 투영했습니다.
  카메라마다 해당 촬영 시각의 q/FK를 사용합니다. 원래 왜곡 영상에 왜곡을 한 번 적용합니다.
  MAP 후보는 혼합 관측 모델을 설명하는 사후 표시이며, 단일 MAP 점으로 재추정한 결과가 아닙니다.
  정답 좌표축은 그리지 않았습니다. 그림 속 좌표축은 각 조건이 추정한 포즈입니다.

ADD 그래프
  figures/R0_ADD_all_frames.png 및 .svg
  figures/R1_ADD_all_frames.png 및 .svg
  figures/R0_R1_ADD_comparison.png 및 .svg
  data/add_by_frame.csv: 프레임, pair ID, 촬영 timestamp, 네 ADD 시계열.
  가로축은 촬영 시간순 1~2,596 프레임입니다. pair ID 000000~002595와의 매핑을 CSV에 기록했습니다.
  세로축은 촬영 시각별 FK 7점 ADD(mm)입니다. 개별 그래프에는 두 카메라를 각각 표시했습니다.
  조건별 두 그래프와 비교 그래프의 y축은 모두 같은 범위입니다.
  모든 원시 샘플을 표시하며 프레임 생략, 평균화, 평활화, 오차 상단 절단을 하지 않았습니다.
  50mm 점선은 참고선입니다. R0/R1은 모두 5,192개 유효 포즈이며 누락 프레임은 없습니다.
  ADD 기준은 기존 기록의 보정 변환과 FK입니다. 독립 외부 계측 GT를 새로 확보한 것은 아닙니다.

정적 비교 예시
  figures/preview_*.jpg: 시작/중간/끝 및 두 카메라 평균 ADD의 최대 개선/악화 사례.
  각 예시는 행=카메라, 열=R0/R1로 배치합니다.
  data/preview_selection.json에 선정 pair와 ADD 차이를 기록합니다.
  이 선택은 시각화용이며 추정이나 하이퍼파라미터 조정에 사용하지 않았습니다.

실제 실행 명령 (작업 디렉터리 ${ROBOTPOSE_ROOT})
  .venv/bin/python -B candidate_background_visualization_20261120/run.py prepare
  .venv/bin/python -B candidate_background_visualization_20261120/run.py render --output smoke --limit 4 --workers 4
  .venv/bin/python -B candidate_background_visualization_20261120/run.py render --output overlays --workers 8
  .venv/bin/python -B candidate_background_visualization_20261120/src/plot_frame_add.py
  .venv/bin/python -B candidate_background_visualization_20261120/src/build_viewer.py
  .venv/bin/python -B candidate_background_visualization_20261120/run.py verify

기존 출력 덮어쓰기를 막기 위해 생성 명령은 이미 결과가 있는 경로에서 실패합니다.
재생성할 때는 같은 Robotpose 루트 아래 새 빈 확장 디렉터리에 run.py와 src/*.py만 복사하여 실행합니다.
prepare 실행 전에 새 확장 안에 data/, figures/, validation/ 디렉터리를 생성하세요.
원본 입력 경로는 candidate_background_ablation_20261116/configs/experiment.yaml에서 읽습니다.
새 실행에도 기존 Python 환경의 NumPy, OpenCV, PyYAML, Matplotlib가 필요합니다.
코드·설정·데이터·가중치·캐시 등 작업 시작 전에 있던 파일을 편집할 필요가 없습니다.

검증
  validation/preparation.json: 10,384개 pose/FK/projection/ADD 전수 검사.
  저장된 pose_projected와 직접 재투영의 최대 차이: 0px.
  좌표축 원점과 FK 0번 점의 최대 차이: 0px.
  기존 ADD CSV와 독립 재계산의 최대 차이: 2.31e-13mm.
  validation/add_graphs.json: 2,596 프레임과 네 곡선의 원시 좌표 일치, 공통 y축, 출력 검증.
  validation/final_verification.json: 개별 이미지 10,384개 디코드/SHA와 기대 출력 집합 일치.
  기존 RGB 5,192개 및 실제 읽은 7개 원본 파일의 전후 SHA/크기/mtime 일치를 검증합니다.
  무관한 다른 데이터셋과 가중치 전체를 다시 해시 검사했다는 뜻은 아닙니다.
  새 코드와 출력, 캐시는 전부 이 확장 디렉터리 안에 있습니다.
