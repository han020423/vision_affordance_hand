# 실험 K: RF-DETR-Seg 비교 실험 (2026-08-20)

## 개명 이력 (2026-08-21)

원래 "실험 J"로 시작했으나 서버에 YOLO v4 미세조정 실험
`custom_finetune_j_grasp_type_v4_seed42`가 이미 존재해 **실험 K로 개명**했다.
사용자 승인(2026-08-21) 하에 다음 이름을 변경했다.

- 설정: `custom_finetune_j_rfdetr_seg.yaml` → `custom_finetune_k_rfdetr_seg.yaml`
- 결과 폴더(로컬·서버): `custom_finetune_j_rfdetr_seg_seed42*` → `custom_finetune_k_rfdetr_seg_seed42*`
- 평가 폴더: `pixel_iou_valid_j_rfdetr*` → `pixel_iou_valid_k_rfdetr*`

완료된 실행 폴더 내부의 `launch_metadata.json`과 데이터셋 manifest는 해시
추적성을 위해 수정하지 않았으므로 이전 id(`custom_finetune_j_rfdetr_seg_seed42`)가
기록에 남아 있다. 아래 본문의 "실험 K"는 모두 이 RF-DETR 실험을 가리킨다.

## 목적

DETR 계열 실시간 모델(RF-DETR-Seg, DINOv2 백본)을 기존 YOLO 계열과 같은 데이터로
미세조정해, "CNN(YOLO11s) + 자체 공개 사전학습" 대 "트랜스포머(DINOv2 사전학습)"
파이프라인을 비교한다. 지도교수의 DINO 계열 경험과도 연결되는 비교 지점이다.

## 비교 설계

| 항목 | 실험 H (기준) | 실험 K (신규) |
|---|---|---|
| 모델 | YOLO11s-seg | RF-DETR-Seg Nano |
| 초기 가중치 | 공개 v2 사전학습(자체 수행) | RF-DETR 공식 COCO 사전학습(DINOv2) |
| 미세조정 데이터 | 자체+공개 v2 replay 혼합 | 동일 원본(아래 COCO 변환본) |
| 입력 해상도 | 640 | 312 (Nano 기본값) |
| 클래스 | handle/body/functional 3클래스 | 동일 |

주의: 초기 가중치와 해상도가 함께 다르므로 이 비교는 "파이프라인 대 파이프라인"이다.
단일 변수 통제 비교가 아니라는 점을 보고서에 명시한다. 공정성 후속 실험이 필요하면
RFDETRSegSmall(해상도 384)로 반복한다.

## 데이터 변환

`scripts/export_rfdetr_coco.py`가 기존 YOLO 변환본을 RF-DETR 관례의 COCO 구조로
변환한다. 원본은 수정하지 않고 하드링크로 연결한다.

- 원본: `yolo_grasp_type_public_v2` + `yolo_grasp_type_custom_v2`
- 출력: `data/processed/rfdetr_mixed_grasp_type_v2` (train/valid/test +
  `_annotations.coco.json`, manifest는 `rfdetr_export_manifest.json`)
- 혼합 규칙은 `build_mixed_finetune_dataset.py`와 동일: 자체 train 10배 반복,
  val은 자체+공개 병합, test는 공개만
- COCO category_id = YOLO class_id + 1 (1=handle, 2=body, 3=functional)
- 배경 음성은 어노테이션 없는 image 항목으로 보존 (train 반복 포함 200장)

2026-08-20 로컬 변환 결과: train 13,884(자체 비중 7.9%) / valid 1,229 / test 1,255,
전량 하드링크, 오버레이 육안 확인 통과(`outputs/rfdetr_check`).

## 실행 방법

설정: `configs/training/custom_finetune_k_rfdetr_seg.yaml`
실행: `scripts/train_rfdetr_seg.py` (launch metadata, run_status, 해시 기록은
train_seg.py와 같은 규칙)

```bash
# 검증만
python scripts/train_rfdetr_seg.py --dry-run
# 1 epoch smoke (수치는 성능으로 보고 금지)
python scripts/train_rfdetr_seg.py --smoke-test
# 전체 학습
python scripts/train_rfdetr_seg.py
```

## 서버 환경 준비 (기존 conda 환경을 건드리지 않는다)

rfdetr 패키지는 기존 `hjh_vision_hand_train` 환경에 설치하지 않는다(의존성 충돌로
Ultralytics 실험 재현성을 해칠 수 있음). 별도 환경을 만든다.

**중요: 서버 드라이버는 525.125.06이라 CUDA 12.1+ 휠이 동작하지 않는다.**
rfdetr를 그냥 설치하면 최신 cu12x torch를 끌고 오므로, cu118 torch를 먼저 고정한다.

```bash
conda create -p /DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_rfdetr python=3.10 -y
conda activate /DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_rfdetr
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install rfdetr
# torch가 cu118로 유지됐는지 반드시 확인한다. 바뀌었으면 위 torch를 재설치한다.
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import rfdetr; print(rfdetr.__version__ if hasattr(rfdetr,'__version__') else 'ok')"
```

데이터는 로컬 변환본을 서버에서 재생성하는 것을 권장한다(공개/자체 YOLO 변환본이
이미 서버에 있으므로 스크립트만 실행하면 된다):

```bash
python scripts/export_rfdetr_coco.py
```

이후 tmux 안에서 smoke → 전체 학습 순서로 실행한다. 전체 학습은 초반 2~3 epoch
정상 여부만 확인하고 분리하며, 완료는 사용자 알림을 기다린다(기존 서버 규칙 동일).

## smoke test에서 확인할 것

1. CUDA 사용 여부(cu118 빌드로 GPU가 실제로 잡히는가)
2. COCO 변환본 로딩(특히 어노테이션 없는 배경 음성 200장에서 로더가 실패하지 않는가)
3. loss가 NaN/Inf가 아닌가
4. `launch_metadata.json`, `run_status.json=completed`, 체크포인트 저장 확인

배경 음성 처리에서 실패하면 converter에 음성 제외 옵션을 추가하기 전에 반드시
사용자와 상의한다(음성 샘플은 배경 오검출 완화 목적으로 넣은 것이므로 무단 제외 금지).

## 평가 계획 (학습 완료 후)

1. 공개 test 1,255장: H의 `best.pt`와 K의 최종 체크포인트를 같은 test로 평가
2. 자체 val(mug_03) 50장: 머그 handle/body 분리 성능 비교
3. container 정책 감사: functional-on-contain 0건 유지 여부
4. 속도: 로컬/서버 추론 지연과 (확보 시) Jetson 실측 — Nano 312 해상도의
   속도 이점과 정확도 손실을 함께 보고한다

mug_04가 없으므로 자체 test 수치는 여전히 보고하지 않는다.

## Jetson 실측 (2026-08-24) — 모델 M

위 "평가 계획" 4번의 Jetson 실측 항목을 채운다. 대상은 실험 K가 아니라 **모델 M**
(`custom_finetune_m_rfdetr_customv3_seed42/checkpoint_best_ema.pth`)이다. 체크포인트
SHA-256이 로컬 원본과 일치함을 전송 후 확인했다(`9b2e23cd…`).

### 측정 환경

- 보드: Jetson Orin Nano Engineering Reference Devkit Super
- OS: Ubuntu 24.04.4, L4T R39.2.1(JetPack 7 계열), CUDA 13.2, 드라이버 595.78
- 작업 경로: `~/hjh_vision_hand` (전용 venv, sudo 미사용)
- 패키지: `torch 2.13.0+cu130`, `torchvision 0.28.0+cu130`, `rfdetr 1.9.3`,
  `opencv-python 4.10.0.84`
- 카메라: USB Sunplus FHD webcam, 1280x720, `conf=0.25`
- GPU 사용 확인: 스크립트가 첫 프레임 후 출력한 `실제 가중치 장치: cuda:0`,
  추론 중 `tegrastats`의 `GR3D_FREQ` 최대 92%

### 속도

| 실행 | 장치 | 평균 추론 | 중앙값 | 유효 FPS | 프레임/시간 |
|---|---|---|---|---|---|
| 로컬 Windows(기존, 참고) | CPU | 261.0 ms | 220.9 ms | 3.83 | 333장 / 100.4초 |
| Jetson 기본 | cuda:0 | 135.4 ms | 130.8 ms | 7.39 | 311장 / 60.1초 |
| Jetson `--optimize` | cuda:0 | 105.0 ms | 97.6 ms | 9.52 | 429장 / 60.1초 |
| Jetson 머그 장면 | cuda:0 | 135.9 ms | 131.5 ms | 7.36 | 309장 / 60.0초 |

엄밀한 비교는 **같은 Jetson 안의 135.4 → 105.0 ms**(중앙값 기준 1.34배)뿐이다.
Windows CPU 열은 기기와 카메라가 달라 참고치이며 동일 조건 비교가 아니다.

`--optimize`(`optimize_for_inference()`)는 성공하지만 그래프 컴파일에 수 분이 걸린다.
상시 구동 파이프라인에는 맞고, 짧게 켰다 끄는 용도에는 맞지 않는다.

### 검출 관찰 (육안, 정량 아님)

머그 60초 실행 결과: `handle` 385건, `body` 309건, `functional` **0건**.

- **handle/body 분리 확인**: `body`가 전 309프레임에서 검출됐고 신뢰도 0.88~0.90.
  손잡이와 몸통이 별도 인스턴스로 나온다.
- **container 정책 유지**: 머그를 60초간 비추는 동안 `functional` 오검출 0건.
- **배경 오검출 없음**: 책상·벽·종이·케이블이 있는 장면에서 배경을
  grasp/functional로 잡는 초기 공개 모델의 문제가 재현되지 않았다.
- 자체 v3에서 추가한 드라이버도 `handle`/`functional` 분리가 동작했다.

### 확인된 결함

1. **handle 마스크 분절**: 385건 / 309프레임 = 프레임당 1.25건으로, 일부 프레임에서
   손잡이가 2개 인스턴스로 쪼개진다. 파지 후보 선택이 분리 컴포넌트를 각각 후보로
   유지하는 구조라 후보 폭 계산이 틀어질 수 있다.
2. **body 마스크가 손잡이 영역 침범**: 인스턴스는 분리돼 있어 정책 위반은 아니지만,
   distance transform 내부점이 손잡이 위치에 생길 수 있다.

두 결함 모두 **육안 관찰이며 수치로 재지 않았다.** 정량화하려면 mug_03 validation
50장에 대해 K·H와 같은 `pixel_iou_summary.json` 프로토콜로 측정해야 한다.

### 아직 하지 않은 것

- 모델 M의 정량 평가(공개 test 1,255장 / 자체 val mug_03 50장). 현재
  `outputs/evaluation/`에는 K와 H만 있고 M은 없다.
- FP16(`model.inference(dtype=torch.float16)`) 속도·정확도 비교.
- 따라서 **M을 배포 모델로 확정할 근거는 아직 없다.**

### 환경 구축 시 주의 두 가지

1. PyTorch 공식 cu130 aarch64 휠은 아키텍처 목록이
   `sm_80, sm_90, sm_100, sm_110, sm_120`이고 Orin의 **sm_87을 명시적으로 제외**해
   미지원 경고를 띄운다. 다만 conv2d·batchnorm·attention 등을 개별 시험한 결과
   전부 PTX JIT로 정상 동작했고 실제 추론도 성공했다. 첫 실행만 JIT 컴파일로 느리다.
   네이티브 sm_87 빌드가 필요하면 `https://pypi.jetson-ai-lab.io/sbsa/cu130`의
   torch 2.11.0을 검토한다.
2. `opencv-python 5.x`는 float32 이미지에 `cv2.putText`를 거부한다
   (`Assertion failed: img.depth() == CV_8U`). `run_realtime_rfdetr.py`의
   `render_overlay`가 float32 버퍼에 그리므로 저장소 고정 버전 4.10.0.84를 써야 한다.
   OpenCV 5로 올릴 계획이면 `render_overlay`를 먼저 고친다.

Jetson 사본의 `scripts/run_realtime_rfdetr.py`에는 `--device` 인자와 실제 가중치 장치
검증을 추가했다(GPU 요청 시 CPU로 떨어지면 중단). 저장소 원본은 수정하지 않았다.

## 모델 M 픽셀 IoU 평가 (2026-08-24, Jetson GPU 실행)

위 "평가 계획" 1~2번을 모델 M에 대해 수행했다. K·H와 **같은 GT**
(`rfdetr_mixed_grasp_type_v2/valid`, conf 0.25, `evaluate_pixel_iou.py`)를 써서
직접 비교 가능하다. customv3의 valid는 v2 valid와 동일하며(1,229장, 자체 50장),
v3 추가분(가위·드라이버)은 train에만 들어갔다.

실행 위치는 Jetson GPU(`~/hjh_vision_hand/.venv`, cuda:0)이고 1,229장에 약 4분
걸렸다. 결과는 `outputs/evaluation/pixel_iou_valid_m_rfdetr/`.

### 자체 val (mug_03, 50장)

| 클래스 | 모델 | mean | median | miss50 | 검출 전무 |
|---|---|---|---|---|---|
| handle (GT 46장) | H(YOLO11s) | 0.5916 | 0.7510 | 8 | 8 |
| | K(RF-DETR v2) | 0.6783 | 0.8826 | 10 | 7 |
| | **M(RF-DETR v3)** | **0.6860** | 0.8570 | 9 | 7 |
| body (GT 50장) | **H(YOLO11s)** | **0.8347** | 0.8837 | **2** | 2 |
| | K(RF-DETR v2) | 0.7239 | **0.9197** | 11 | **1** |
| | M(RF-DETR v3) | 0.7119 | 0.7701 | 10 | 2 |

머그에는 functional GT가 0장이다(container 정책대로 컵 몸통을 functional로 만들지
않음). Jetson 실시간 시험에서 관측한 `functional` 0건은 이 GT와 일치하는 정상 동작이다.

### 공개 val (1,179장)

| 클래스 | 모델 | mean | median | miss50 | 검출 전무 |
|---|---|---|---|---|---|
| handle (GT 912장) | H(YOLO11s) | 0.5859 | 0.7023 | 174 | 164 |
| | K(RF-DETR v2) | **0.7177** | **0.7585** | 34 | 16 |
| | **M(RF-DETR v3)** | 0.7113 | 0.7472 | **27** | **12** |
| body (GT 165장) | H(YOLO11s) | 0.8424 | 0.8685 | 4 | 0 |
| | K(RF-DETR v2) | **0.8610** | **0.9151** | 4 | 0 |
| | M(RF-DETR v3) | 0.8581 | 0.9124 | 4 | 0 |
| functional (GT 881장) | H(YOLO11s) | 0.7525 | 0.8387 | **49** | **45** |
| | K(RF-DETR v2) | 0.7586 | 0.8628 | 64 | 62 |
| | **M(RF-DETR v3)** | **0.7708** | **0.8666** | 52 | 46 |

### 해석

1. **v3 추가는 공개 성능을 해치지 않았고 일부 개선했다.** M은 K 대비 공개 handle
   검출 전무 16→12, miss50 34→27로 줄었고, functional은 mean 0.7586→0.7708,
   검출 전무 62→46, miss50 64→52로 K의 약점이던 functional recall이 H 수준으로
   회복됐다. body는 사실상 동률이다. 가위·드라이버를 넣어도 기존 과제가 퇴화하지
   않음을 확인했다.
2. **자체 머그 body는 M이 H보다 확실히 나쁘다.** mean 0.7119 대 0.8347,
   miss50 10 대 2다. K에서 M으로 오며 median이 0.9197→0.7701로 크게 떨어졌는데,
   mean은 거의 그대로이므로 잘 되던 사례가 나빠진 쪽이다.
3. **공개 handle에서는 H가 치명적이다.** 검출 전무 164/912(18.0%)로, M의
   12/912(1.3%)와 비교가 되지 않는다. 일반화 측면에서는 RF-DETR 계열이 낫다.
4. **세 모델 모두 머그 손잡이를 46장 중 7~8장(15~17%)에서 통째로 놓친다.**
   이는 배포 모델 선택과 무관한 공통 결함이며, 손잡이 파지가 목표인 파이프라인에서
   가장 큰 위험이다. 2026-08-24 Jetson 실시간 시험에서 관측한 손잡이 마스크 분절과
   같은 현상으로 보인다.

### 배포 모델 결론

**단일 최고 모델이 없다.** M은 RF-DETR 계열 중 최선이고 공개 일반화에서 H를 크게
앞서지만, 자체 머그 body에서는 H가 앞선다. 따라서 이 수치만으로 배포 모델을
확정하지 않는다. 결정은 사용자가 우선순위(자체 머그 정밀도 대 일반 물체 일반화)를
정한 뒤에 내린다.

어떤 모델을 고르든 4번의 손잡이 검출 전무 15~17%를 먼저 다루어야 한다.
mug_04를 아직 쓰지 않았으므로 자체 test 수치는 여전히 보고하지 않는다.

## 모델 M 신뢰도 임계값 스윕 (2026-08-24, Jetson GPU)

위 평가에서 드러난 "머그 손잡이 검출 전무 7/46"이 임계값 문제인지 모델 문제인지
가리기 위해 conf 0.15/0.25/0.50을 같은 GT로 측정했다. 결과 폴더는
`outputs/evaluation/pixel_iou_valid_m_rfdetr_conf015`, `_conf050`.

### M: 자체 val (mug_03, 50장)

| conf | handle mean | handle 전무 | handle miss50 | body mean | body 전무 | body miss50 |
|---|---|---|---|---|---|---|
| **0.15** | **0.7032** | **4** | **7** | **0.7385** | **1** | **9** |
| 0.25 | 0.6860 | 7 | 9 | 0.7119 | 2 | 10 |
| 0.50 | 0.6774 | 10 | 11 | 0.5341 | 15 | 19 |

### M: 공개 val (1,179장)

| conf | handle mean | handle 전무 | handle miss50 | functional mean | functional 전무 |
|---|---|---|---|---|---|
| **0.15** | 0.6988 | **0** | **15** | **0.7954** | **14** |
| 0.25 | **0.7113** | 12 | 27 | 0.7708 | 46 |
| 0.50 | 0.5951 | 169 | 186 | 0.7359 | 103 |

공개 body는 세 임계값 모두 mean 0.8581 / 전무 0 / miss50 4로 동일하다.

### 해석

1. **M에서 손잡이 미검출은 상당 부분 임계값 문제였다.** conf 0.25→0.15에서 머그
   손잡이 검출 전무가 7→4로, 공개 손잡이 검출 전무가 12→**0**으로 줄었다.
   functional 검출 전무도 46→14로 개선됐다. 공개 body는 전혀 영향받지 않았다.
2. **K는 같은 조정에 반응하지 않았다.** K는 0.25→0.15에서 머그 손잡이 검출 전무가
   7로 그대로였다(위 K 스윕 표). 즉 v3 추가 학습이 M의 신뢰도 분포를 바꿔
   낮은 임계값에서 회수 가능한 예측을 만들어 냈다. 이는 v3 데이터 추가의
   부수 효과이며 K→M의 실질적 이득이다.
3. **conf 0.50은 쓰면 안 된다.** 공개 손잡이 검출 전무 169건, 머그 body 검출 전무
   15건으로 무너진다. `configs/hardware/realtime_camera.yaml`의 기본값이 0.5이므로
   M을 실시간에 쓸 때 반드시 낮춰야 한다. 2026-08-24 Jetson 실시간 시험은 CLI
   `--conf 0.25`로 덮어써서 실행했다.

### 세 모델 최적 임계값 비교

각 모델을 자기에게 가장 유리한 conf 0.15로 맞춘 비교다.

| 지표 | H@0.15 (YOLO11s) | K@0.15 (RF-DETR v2) | **M@0.15 (RF-DETR v3)** |
|---|---|---|---|
| 머그 handle mean | 0.5567 | 0.6742 | **0.7032** |
| 머그 handle 전무 | 6 | 7 | **4** |
| 머그 body mean | **0.8274** | 0.7220 | 0.7385 |
| 머그 body miss50 | **2** | 11 | 9 |
| 공개 handle 전무 | 110 | — | **0** |
| 공개 functional mean | 0.7642 | — | **0.7954** |
| 공개 functional 전무 | 25 | — | **14** |

**M@0.15가 머그 손잡이와 공개 전 지표에서 최선이다.** 남은 단일 약점은 머그 body로,
H가 mean 0.8274 / miss50 2인 반면 M은 0.7385 / 9다. 머그 몸통 POWER 파지의
정밀도가 중요하면 이 격차를 별도로 다뤄야 한다.

mug_04를 쓰지 않았으므로 자체 test 수치는 여전히 보고하지 않는다.

## 실험 N: robot_hand 클래스 추가 (customv4, 2026-08-25)

로봇손이 파지 후보로 오검출되는 문제의 근본 해결로, 분할 모델에 `robot_hand`(category 4)를
추가했다. 데이터는 젯슨 실환경 손 영상 3개에서 뽑은 프레임 198장을 GroundingDINO+SAM2로
의사라벨하고 사람이 검수(승인 155: train 116, valid 39)했다. 머그 동반 프레임에는 모델 M의
handle/body 예측 101개를 병합했고, functional 의사라벨 9건은 전부 손·장비 오검출이라 제외했다.
데이터셋 `rfdetr_mixed_grasp_type_customv4`(v3 하드링크 승계, 기존 라벨 불변),
설정 `configs/training/custom_finetune_n_rfdetr_customv4_hand.yaml`(M과 동일 레시피).

학습: 2026-08-24 22:28 ~ 01:11 (2시간 43분), 15/50 epoch에서 early stopping, 최고 mAP 0.608.

### robot_hand 픽셀 IoU (v4 valid 손 프레임 39장, conf 0.25, 젯슨)

39/39 검출(재현율 1.0), mean 0.9191 / median 0.9537 / min 0.6849.
GT가 SAM2 의사라벨의 사람 승인본이므로 완전 수동 GT보다 관대한 기준임을 명시한다.
결과: `outputs/evaluation/hand_iou_valid_n/summary.json` (Jetson).

### 기존 3클래스 회귀 확인 — M과 같은 GT·프로토콜 (`pixel_iou_valid_n_rfdetr`)

| subset | 클래스 | M mean/med (전무) | N mean/med (전무) |
|---|---|---|---|
| custom(mug_03 50장) | handle | 0.6860/0.8570 (7) | 0.6365/0.8913 (**11**) |
| | body | 0.7119/0.7701 (2) | **0.7341/0.8978** (1) |
| public(1,179장) | handle | 0.7113/0.7472 (12) | 0.7115/0.7543 (12) |
| | body | 0.8581/0.9124 (0) | 0.8610/0.9101 (0) |
| | functional | 0.7708/0.8666 (46) | 0.7564/0.8630 (54) |

판정: public은 사실상 동일, custom body는 개선(median +0.13). 유일한 악화는
**custom handle의 완전 미검출 7→11장**(마스크가 잡힐 때 품질은 median 기준 오히려 좋아짐)과
public functional 완전 미검출 +8. 손 클래스 이득(0.92 IoU, 오검출 근본 해결) 대비 수용 가능한
수준이나, 실전에서 손잡이 검출률을 관찰할 것.

배포: 체크포인트를 젯슨 동일 경로에 복사, `run_realtime_rfdetr.py`에 robot_hand 통합
(`--hand-source segmentation`: robot_hand 마스크 → 손 위치·접근 방향, 후보에서 제외,
억제 필터 박스 연동, 구/신 모델 겸용 class base 판정).

## 실험 O: v5 (N 실패 복구 — Copy-Paste 합성) 결과 (2026-08-25)

15/50 epoch 조기종료, 최고 EMA mAP 0.6243(epoch 4, 상호작용 합성 120장이 포함된 v5 valid 기준).
epoch 4 클래스별: handle 0.427(P 0.87/R 0.68), body 0.629(P 0.88/R 0.82), functional 0.512,
robot_hand 0.947(F1·P·R 모두 1.00).

### 픽셀 IoU 3자 비교 (공통 GT·conf 0.25, `pixel_iou_valid_o_rfdetr`)

| subset | 클래스 | M mean/med (전무) | N mean/med (전무) | O mean/med (전무) |
|---|---|---|---|---|
| custom(mug_03) | handle | 0.6860/0.8570 (7) | 0.6365/0.8913 (11) | 0.6443/0.7628 (**6**) |
| | body | 0.7119/0.7701 (2) | 0.7341/0.8978 (1) | **0.7808/0.9501** (2) |
| public | handle | 0.7113/0.7472 (12) | 0.7115/0.7543 (12) | **0.7183/0.7559** (**9**) |
| | body | 0.8581/0.9124 (0) | 0.8610/0.9101 (0) | **0.8623/0.9131** (0) |
| | functional | 0.7708/0.8666 (46) | 0.7564/0.8630 (54) | 0.7497/0.8674 (**80**) |

robot_hand IoU(손 프레임 39장, GT=승인 의사라벨): 39/39 검출, mean 0.9405/med 0.9521,
최저 0.8347 (N 최저 0.6849 대비 일관성 개선).

판정: N의 회귀였던 custom handle 완전 미검출이 11→6으로 M(7)보다도 줄었고, body는 전 구간
최고다. 유일한 악화는 public functional 완전 미검출 46→80(881장 중, 9.1%)로, 합성에서
가려진 functional 라벨을 가시 영역 30% 미만 시 제거한 규칙의 부작용일 수 있다. 자체 도메인
머그에는 functional GT가 없어 시연 영향은 제한적이나 명시해 둔다. 최종 채택은 실물 상호작용
시험(M과 나란히 비교)을 통과해야 한다.

## 실험 P: v6 (2026-08-25 신규 촬영 4종) 결과 (2026-08-26)

O의 실물 한계 세 가지(실제 접근 자세의 robot_hand 미검출, 원통 물체의 handle 오인,
검정 물체의 robot_hand 오인)를 겨냥해 신규 촬영 4종을 반영했다.

데이터 v6 (`build_hand_dataset_v6.py`, 2단계):

- core: v4 − handle 라벨 누락 머그 프레임 20장(v5와 같은 규칙)
  + train: 접근 자세 손 112장(robot_hand), 원통 286장(전부 body — 무손잡이 물체),
    검정 머그·텀블러 80장(robot_hand 네거티브 겸용)
  + valid: 실사 상호작용 56장 — **손 검수·물체 검수 모두 승인된 프레임만**
    (한쪽만 승인되면 보이는 무라벨 물체가 거짓 음성이 되므로 제외, v4 교훈)
- 합성: 구 합성을 승계하지 않고 전체 손 조각 풀(구 155 + 신 217)로 2,200/120장
  재생성(`build_hand_copy_paste.py` 확장 — 원통·검정 프레임도 바탕에 포함).
  개방 손바닥 편중 제거가 목적이다.
- 최종: train 17,468 / valid 1,444 / test 1,255(불변)
- 의사라벨 정리(`clean_v6_object_labels.py`): 고정 배경(콘센트·파워서플라이) functional
  989건 자동 제거, 손 겹침 인스턴스 275건 제거, 원통 handle→body 121건 교정.
  사람 검수: 손 217/339, 물체 427/485 승인.

학습: 15/50 epoch 조기종료, 최고 EMA mAP **0.6310**(epoch 4). O(0.6243)보다 높지만
valid 구성이 실험마다 달라(실사 상호작용 56장 추가) mAP의 실험 간 직접 비교는 하지 않는다.

### 픽셀 IoU 3자 비교 (공통 GT·conf 0.25, `pixel_iou_valid_p_rfdetr`)

| subset | 클래스 | M mean/med (전무) | O mean/med (전무) | P mean/med (전무) |
|---|---|---|---|---|
| custom(mug_03) | handle | 0.686/0.857 (7) | 0.644/0.763 (6) | **0.700/0.859** (**5**) |
| | body | 0.712/0.770 (2) | 0.781/0.950 (2) | **0.866/0.966** (**1**) |
| public | handle | 0.711/0.747 (12) | 0.718/0.756 (9) | 0.715/0.749 (13) |
| | body | 0.858/0.912 (0) | 0.862/0.913 (0) | 0.854/0.908 (0) |
| | functional | 0.771/0.867 (46) | 0.750/0.867 (80) | **0.775/0.867** (**41**) |

O의 유일한 회귀였던 public functional 완전 미검출이 80→41로 해소됐고 M(46)보다도 좋다.
custom handle·body는 전 구간 최고. public handle·body는 동급이다.

### robot_hand IoU (GT = 승인 의사라벨, conf 0.25)

- 구 valid(212955, 39장): 39/39 검출, mean 0.9408/med 0.9531, 최저 0.7824 — O와 동급
- **접근 자세**(approach_valid, 105장): 미검출 P 2장 / O 5장, median P 0.898 / O 0.924,
  mean P 0.781 / O 0.841. mean 열세는 P가 손 아래 배선까지 robot_hand에 포함하는
  과분할 때문임을 시각 확인했다(GT는 손만). 실전에서는 배선 영역 오검출까지 함께
  억제되므로 무해~유익하다. 이 GT 영상의 실사 56장이 P의 valid에 포함되므로 조기 종료
  선택이 P에 유리했을 수 있음을 명시한다.

### v6 표적 항목 검증 (`evaluate_v6_targets.py`; 원통·검정은 P의 학습 프레임이므로 P 수치는 낙관적 상한)

| 항목 | O | P |
|---|---|---|
| 원통 286장: body 완전 미검출 | 83 | **0** |
| 원통 286장: handle 오검출 프레임 | 120 (42%) | **6** |
| 원통 body IoU (mean/med) | 0.634/0.880 | 0.921/0.961 |
| 검정 80장: robot_hand 오검출 (conf 0.25/0.5) | 0 / 0 | 0 / 0 |
| 검정 handle / body IoU | 0.839 / 0.964 | 0.870 / 0.968 |

검정 물체의 robot_hand 오인은 오프라인 정지 프레임에서는 O도 재현되지 않아(실물에서는
손 동반·특정 각도에서 발생) 이 항목의 개선은 오프라인으로 입증되지 않는다 — 실물
시험에서 확인했다(아래 base 버그 참조: 상당 부분이 파이프라인 버그였다).

## 실시간 클래스 기준(base) 버그 사후 분석 (2026-08-26)

**증상**: P를 실물 시험하자 학습 프레임과 같은 조건의 종이컵·캔이 handle 0.9로 표시됐다.
오프라인 평가(전 항목 정상)와 라이브만 다르다는 모순이 단서였다.

**원인**: 실시간 스크립트가 클래스 번호 기준(0시작/1시작)을 **매 프레임** "최소 id==0이면
0시작, 아니면 1시작"으로 재추정했다. 우리 모델은 0시작(0=handle, 1=body, 2=functional,
3=robot_hand)이므로, handle(0) 검출이 없는 프레임은 전부 1시작으로 오판되어 **모든
클래스가 한 칸씩 밀렸다**: body→handle 표시·판정, robot_hand→functional(손 미검출 처리).

**파급**: 실험 N 배포 이래 라이브에서 관찰된 "손이 functional로 뜬다", "원통이 handle",
"클래스가 프레임마다 요동"의 상당 부분이 모델이 아니라 이 시프트였다. 머그 장면만 늘
정상이었던 이유(손잡이 id 0이 거의 항상 검출되어 기준이 우연히 맞음)도 설명된다.
오프라인 평가는 기준을 0으로 고정해 버그를 우회했기에 지표-실물 괴리로 나타났다.

**수정**: 확정 증거가 나올 때까지 실측 기본값(0시작)을 쓰고, 증거(id 0 관측 = 0시작 /
0시작이면 불가능한 id 4 관측 = 1시작)가 나오면 잠근다. 종이컵 프레임 종단 검증으로
body 4건/handle 0건을 확인했다.

**교훈**: 프레임 내용에 따라 달라지는 값(검출된 클래스 집합)으로 불변 속성(모델의 번호
체계)을 추정하면 안 된다. 지표와 실물이 다르면 코드 경로 차이부터 의심한다.

## 부위 수준 파지 결정 — 파지점 폐지 (2026-08-26)

base 버그 수정 후에도 남던 "손 근접 시 파지점 소실"에 대해, 파지점 자체를 판정에서
빼는 설계 변경(사용자 제안)을 채택했다.

**근거**: 이 시스템의 공유 제어 구조에서 조준은 사람이 하고(손 접근 = 의도 표현) 구동은
접촉(스톨) 적응이 담당하므로, 픽셀 파지점은 구동에 쓰이지 않으면서 가림에 가장 취약한
산출물이었다. 반자율 의수 문헌(Došen et al. 2010, Marković et al. 2014)도 비전은 파지
**유형·개구**를 결정하고 접촉점은 정하지 않는다.

**구현** (`src/grasp_selection/part_scoring.py`, `scripts/run_realtime_rfdetr_part.py`;
점 기반 경로는 보존해 A/B 비교):

- 자세 구조 규칙·인접 타이브레이크는 기존 scoring과 공유. 접근 방향 정렬은 파지점 대신
  부위 마스크의 손 최근접 픽셀 기준.
- 기능부 회피는 점-근접 대신 **부위 얽힘 비율**: 부위 픽셀의 50% 초과가 기능부 하한
  거리(12px) 안이면 거부. 도구 자루처럼 날과 맞닿은 부위는 통과, 날을 따라 붙은 조각만
  거부된다.
- **유령 후보**(hand_filter): 검증(순위 도달)을 통과한 후보의 스냅샷을 저장해 두고,
  후보가 소멸하면(모델 미검출·손 픽셀 통제거·손 박스 거부 — 근접 가림의 세 증상) 대신
  내보낸다. 손 박스가 그 자리를 덮는 동안 TTL 동결(호버 무기한), 빈자리 ~2.4초, 다른
  안정 후보가 스냅샷의 30% 이상을 덮으면(교체 증거) 즉시 폐기, 하드캡 ~23초. 실측
  후보가 있으면 점수 할인(×0.85)으로 즉시 양보한다. 표시는 1위일 때만 회색 HOLD.
- 계측: 결정 단계 12.8ms(점 기반 20.3ms), 접근 원본 영상 792프레임 리플레이에서
  접근·가림 중 선택 소실 0프레임(`replay_selection_pipeline.py`).

**실물 확인**: 부위 강조가 점보다 안정적임을 사용자가 확인했고, P + 부위 결정 + 손 제어
(스페이스 트리거) 조합으로 실물 파지 성공. 이 구성을 기본 배포로 확정했다.
