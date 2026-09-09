# 분리형 손 검출기 실험 (방법 ③, 2026-08-25)

## 배경

실험 N(분할 모델에 robot_hand 클래스 통합)은 실전에서 물체 인식 붕괴로 롤백됐다.
본질은 "손을 가르치다 물체가 망가짐"이므로, 방법 ③은 물체 모델(M)을 일절 건드리지
않고 **1클래스 초경량 손 검출기(YOLO11n detect)** 를 별도로 둔다. 손 박스만 있으면
후보 억제 필터와 손 위치·접근 방향 추적, mm/px 추정이 모두 동작하므로 분할이 필요
없다. 모델이 분리되어 물체 성능 오염 경로가 구조적으로 없다.

## 데이터셋 `yolo_hand_detector_v1` (서버)

v5(rfdetr_mixed_grasp_type_customv5) COCO 주석에서 robot_hand 박스만 추출해 재사용
(`scripts/build_hand_detector_dataset.py`, 원본 하드링크·읽기 전용).

| 분할 | 양성(실제) | 양성(합성) | 음성(custom) | 음성(공개) |
|---|---:|---:|---:|---:|
| train | 96 | 2,200 | 191 | 800 |
| valid | 39 | 120 | 50 | 200 |

- 실제 = 젯슨 손 영상 3개에서 사람이 검수·승인한 프레임. 합성 = Copy-Paste
  손-물체 겹침(상호작용 강건성). 분할은 v5 승계(영상 단위, crop 누수 방지 유지).
- 음성은 빈 라벨(custom 물체·배경 전부 + 공개 표본, replay 중복 제거 `__r\d+|__rep\d+`).
- 라벨 육안 검증: 의수만 타이트하게, 잡은 사람 손가락 제외 확인.

## 학습 (`scripts/train_hand_detector.py`, 서버 GPU 2)

YOLO11n(detect) 사전학습 가중치, 60 epoch, imgsz 640, batch 32, seed 42,
patience 15, **좌우/상하 반전 증강 금지**(의수 좌우 고정 — Copy-Paste 생성과 같은 근거).
출력: `outputs/training/hand_detector_v1_yolo11n_seed42` (서버),
로컬 `models/hand_detector/hand_detector_v1_best.pt` (5.5MB).

## 결과

### valid (409장, 손 159 인스턴스)

P 0.988 / R 0.999 / mAP50 0.994 / mAP50-95 0.953. 추론 0.5ms(서버 GPU).

conf 스윕(0.25/0.40/0.50 모두 동일): **음성 오검출 0/250장, 실제 손 미검출 0/39장**,
검출 평균 conf 0.871. 분포 내에서는 임계값에 둔감하다.

### 교차 도메인: 로컬 카메라 영상 3편 (`track_hand_video.py`, backend custom)

| 영상 | 검출률 | 판정 |
|---|---:|---|
| 15_37_07 (탑다운·검은 책상) | 0.0% | conf 0.02에서도 0 — 미학습 시점, 완전 실패 |
| 16_10_54 (낮은 각도·잡동사니) | 0.0% | 〃 |
| 16_22_36 (흰 배경·사선 = 시연 후보 배치) | 72.8% | **미검출 프레임 전수 확인 결과 손이 화면에 없는 구간** — 손이 보이는 동안은 사실상 전부 검출, 파지 상호작용 구간 포함 |

### motion 백엔드 대비 (시연 배치 기준)

| | motion(배경 차분) | 전용 검출기 |
|---|---|---|
| 16_22_36 검출률 | 59% | 72.8% (가시 구간 전부) |
| 정지한 손 | 불가(원리적) | 가능 |
| 사람 동시 등장 | 실패(20%) | 학습 도메인이면 무관 |
| mm/px 추정 | 덩어리 폭(불안정) | 박스 폭(안정) |

## 한계와 후속

- **학습 시점 밖에서는 0%.** 손 데이터가 젯슨 카메라·흰 천 배경 단일 세션이라
  탑다운·잡동사니 배경을 전혀 일반화하지 못한다. 시연 배치(흰 배경·사선)와 젯슨
  도메인에서는 충분하나, 다른 배치가 필요하면 해당 시점 프레임을 추가해야 한다
  (기존 GroundingDINO+SAM2 의사라벨 → 사람 검수 → 재학습 경로 재사용, v2).
- 닫힌 손(파지 완료 상태) 프레임은 학습에 적고, 영상 3에서는 문제로 나타나지
  않았으나 실물 확인이 남아 있다.

## 통합

`src/perception/hand_tracker.py`의 backend=custom이 기존 구현 그대로 사용됨(코드
변경 없음). 전환은 설정 파일 하나:
`configs/hardware/hand_tracker_detector.yaml` (conf 0.40, detect_every 3(CPU)/1(GPU),
hand_width_mm 120 실측값으로 mm/px 추정).

```bash
python scripts/run_realtime_seg.py --select --hand-source detector \
  --hand-tracker-config configs/hardware/hand_tracker_detector.yaml
```

젯슨 배포 시: ultralytics 설치(venv pip) + 가중치 복사. RF-DETR 런타임
(`run_realtime_rfdetr.py`, 다른 세션 관리)에 붙일 때는 해당 세션과 조율한다.

## 결합 파이프라인 검증 (M + 손 검출기, 2026-08-25)

`scripts/run_grasp_pipeline_video.py`: 실시간 프로그램과 같은 흐름(M 분할 → 손 추적 →
억제 필터 → 후보 선택)을 녹화 영상으로 재현하는 오프라인 검증 프로그램. 실시간
프로그램의 헬퍼(render_overlay, detections_to_arrays)를 재사용한다.

시연 배치 영상(16_22_36, 492프레임, 로컬 CPU) 결과:

| 항목 | 값 |
|---|---|
| 손 추적률 | 72.8% (손이 보이는 구간 사실상 전부) |
| 상태 분포 | GRASP 274 / ALIGN 65 / NO_TARGET 19 / NO_HAND 134(손 부재) |
| GRASP 중 자세 | POWER 272 (머그 몸통), PRECISION 2 (일시적) |
| GRASP 점수 | 평균 0.784 (0.630~0.904) |
| 추론 | M 254ms/프레임 (CPU; 젯슨 GPU에서는 대폭 단축) |

핵심 관찰: **M이 움직이는 손을 handle/functional로 오검출하는 프레임에서도, 손
검출기의 박스가 억제 필터(hand_overlap 거부)로 해당 후보를 걸러내 결정이 머그에
유지된다.** 실험 N에서 발생한 "손 진입 시 인식 붕괴"가 모델 분리 구조에서는 후보
선택 단계에서 차단됨을 확인했다. 손이 화면 밖이면 NO_HAND로 결정을 중단한다
(오래된 손 위치로 파지하지 않는 안전 규칙).

실시간(카메라) 결합 실행은 기존 프로그램에 설정만 바꿔 쓴다:

```bash
python scripts/run_realtime_rfdetr.py --select --hand-source detector \
  --hand-tracker-config configs/hardware/hand_tracker_detector.yaml
```

(--hand-mock 또는 --hand-port를 주면 손 제어까지 연결된다. 코드 변경 없음.)

## 젯슨 배포 (2026-08-25)

- ultralytics 8.3.163을 `--no-deps`로 설치(+pandas·psutil·py-cpuinfo·thop)해
  **OpenCV 4.10 고정을 보존**했다. 설치 후 cv2 4.10.0·CUDA 사용 가능 확인.
- 업로드: 검출기 가중치(5.5MB), `hand_tracker_detector_jetson.yaml`(cuda:0,
  매 프레임 검출), `run_grasp_pipeline_video.py`, 갱신된 `hand_tracker.py`
  (모션 디버그 계측 추가분 — 나머지 소스는 md5 일치 확인).
- GPU 검증: 젯슨 손 영상 212955(207프레임, 손 학습의 valid 영상)로 결합
  파이프라인 실행 — **손 추적률 91.3%, 결합 추론 135.8ms/프레임(약 7.4 FPS)**.
  sm_87 경고는 기존 확인대로 무시 가능.

### 젯슨 실행 명령

```bash
# 오프라인 검증 (영상 파일)
python scripts/run_grasp_pipeline_video.py --video <영상> \
  --hand-tracker-config configs/hardware/hand_tracker_detector_jetson.yaml

# 실시간 (카메라, 손 제어는 --hand-mock/--hand-port 추가)
python scripts/run_realtime_rfdetr.py --select --hand-source detector \
  --hand-tracker-config configs/hardware/hand_tracker_detector_jetson.yaml
```

### 젯슨 검증에서 드러난 남은 문제

물체가 없는 장면에서 **왼쪽 가장자리의 실험 장비를 M이 body 0.56으로 오검출**해
GRASP 21프레임이 발생했다(손 오검출은 손 박스 억제로 차단되지만, 장비는 정지
물체라 시간 안정성 필터도 통과한다). 완화 선택지:
1. 시연 배치에서 장비·케이블을 화면 밖으로 (환경 통제 — 기존 계획과 동일)
2. body 후보의 최소 신뢰도 상향(0.35→0.6): 실측 머그 body는 0.68~0.90이라 여유 있음
3. 프레임 경계에 걸린(touches_boundary) 후보 거부 규칙 추가
어느 것이든 사용자 승인 후 적용한다.
