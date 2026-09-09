# Claude Code 상세 인수인계

작성 기준 시각: 2026-08-18 (Asia/Seoul)

이 문서는 현재 로컬 저장소와 학습 서버에서 확인한 실제 상태를 기준으로 작성했다.
서버·장치 상태는 바뀔 수 있으므로 새로운 작업을 시작할 때 읽기 전용 명령으로 다시
확인한다.

---

## 1. 한 줄 현재 상태

Aff-Grasp와 UMD 공개 데이터 사전학습, `contain` 정책 문제 분석과 개선 학습, 자체 머그
데이터 140장 수동 검수, 배경 20장 포함 승인본 생성, YOLO 변환, 서버 배치 및 A/D
미세조정 설정 dry-run까지 완료했다. **자체 데이터 A/D 전체 미세조정은 아직 시작하지
않았고, `mug_04`가 없으므로 자체 test 평가는 아직 할 수 없다.**

---

## 2. 절대 변경하면 안 되는 결정

자세한 내용은 `AGENTS.md`와 `PROJECT_CONTEXT.md`를 우선한다.

### 의미 라벨

```yaml
classes:
  0: grasp_region
  1: functional_region
ignore_index: 255
```

중간 semantic PNG 저장값은 다음과 같다.

| PNG 값 | 의미 | 모델 클래스 ID |
|---:|---|---:|
| 0 | background | 없음 |
| 1 | grasp_region | 0 |
| 2 | functional_region | 1 |
| 255 | ignore | 없음 |

### 공개 데이터 매핑

- Aff-Grasp `graspable → grasp_region`
- Aff-Grasp `functional → functional_region`
- UMD `grasp`, `wrap-grasp → grasp_region`
- UMD `cut`, `scoop`, `pound`, `support → functional_region`
- UMD `contain → ignore`

`contain` 통합 PNG를 `functional_region`이나 background로 다시 저장하면 안 된다.
다만 현재 선택한 공개 데이터 비교 실험은 **통합 PNG를 바꾸지 않고**, YOLO 변환본에서
`contain` 폴리곤을 기록하지 않아 암묵적 배경으로 학습했다. 이 예외는
`docs/contain_background_experiment.md`에 기록돼 있다.

### 자체 머그 정책과 분할

- 손잡이와 몸통은 둘 다 `grasp_region`이다.
- 손잡이와 몸통은 서로 다른 인스턴스 ID로 유지한다.
  - instance 1: handle
  - instance 2: body
- 자체 머그에는 임의로 `functional_region`을 만들지 않는다.
- `mug_01`, `mug_02`: train
- `mug_03`: validation
- `mug_04`: 촬영 예정인 unseen-instance test
- 배경 음성 샘플 20장: train
- 프레임 단위 random split을 사용하지 않는다.

---

## 3. 로컬과 서버 환경

### 로컬 Windows

- 저장소: `C:\Users\han02\Documents\sail\vision_hand`
- Python: 3.12.6
- PyTorch: 2.7.1 CPU 빌드
- Ultralytics: 8.3.163
- 로컬 CUDA: 사용 불가
- `.git` 폴더: 없음. 현재 폴더는 Git 저장소로 인식되지 않는다.
- 로컬은 변환, 검증, 오버레이 확인과 문서 작업에 사용한다.
- 실제 학습은 서버에서 수행한다.

Ultralytics를 직접 import하면 사용자 AppData에 설정을 쓰려다 권한 경고가 날 수 있다.
`scripts/train_seg.py`는 import 전에 저장소 내부 `.ultralytics`를 지정하도록 수정돼
있으므로 가능한 한 이 스크립트를 사용한다.

### 학습 서버

- 접속: 호스트·포트·계정은 공개 저장소에 두지 않는다. 연구실 담당자에게 별도로 받는다.
- 인증: SSH key를 쓴다. 비밀번호는 문서·코드·로그 어디에도 저장하지 않는다.
- 작업 폴더: 계정 홈의 `hjh_vision_hand`
- tmux 세션: `5`
- 마지막 사용 창: `custom_setup`
- conda 환경 경로:
  `/DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_vision_hand_train`
- Python: 3.10.20
- PyTorch: 2.7.1+cu118
- Ultralytics: 8.3.163
- GPU: NVIDIA GeForce RTX 3090 24GB급 4장
- 확인 당시 드라이버: 525.125.06

서버 저장 경로는 다음 symlink로 구성돼 있다.

```text
~/hjh_vision_hand/data
  -> /DATA/<계정>/hjh_vision_hand_storage/releases/vision_hand_server_training/data

~/hjh_vision_hand/outputs
  -> /DATA/<계정>/hjh_vision_hand_storage/outputs
```

따라서 데이터와 학습 결과는 홈이 아니라 `/DATA`에 저장된다. 확인 당시 `/DATA` 여유는
약 1.5TB였지만 다음 작업 전에 `df -h /DATA`로 다시 확인한다.

서버에서 반드시 지킬 사항:

- `tmux` 안에서 작업한다.
- conda 환경을 사용한다.
- `sudo`를 사용하지 않는다.
- apt, 드라이버, CUDA, 서버 OS를 업데이트하지 않는다.
- 큰 파일을 `~`에 새로 쌓지 않는다.

---

## 4. 지금까지 완료한 공개 데이터 작업

### 4.1 다운로드와 원본 보존

- Aff-Grasp 원본: `data/raw/affgrasp`
- UMD 원본: `data/raw/umd`
- 원본 파일은 변환 과정에서 수정하지 않았다.
- AED는 학습에 사용하지 않고 외부 평가용으로 유지했다.

### 4.2 주요 구현

| 파일 | 역할 |
|---|---|
| `src/labeling/policy.py` | 고정 라벨 매핑과 저장값 정책 |
| `src/datasets/affgrasp.py` | Aff-Grasp 파서와 통합 라벨 변환 |
| `src/datasets/umd.py` | UMD 수동 GT 파서와 통합 라벨 변환 |
| `src/datasets/splits.py` | 물체 ID 기반 UMD split 검사 |
| `src/datasets/yolo_export.py` | 통합 mask를 감사 가능한 YOLO polygon으로 변환 |
| `scripts/prepare_affgrasp.py` | Aff-Grasp 변환 CLI |
| `scripts/generate_umd_splits.py` | UMD 물체 ID split 생성 |
| `scripts/prepare_umd.py` | UMD 변환·서브샘플 CLI |
| `scripts/validate_dataset.py` | 통합 manifest, mask, split 누수 검사 |
| `scripts/render_manifest_overlays.py` | 사람이 볼 overlay 생성 |
| `scripts/export_yolo_public.py` | YOLO 데이터셋 export |
| `scripts/validate_yolo_dataset.py` | YOLO 파일·좌표·manifest·split 검사 |

### 4.3 공개 YOLO 데이터 수량

엄격한 기본 변환본 `data/processed/yolo_public`:

- 원본 manifest 레코드: 29,174
- train: 2,134
- validation: 470
- test: 494
- grasp 인스턴스: 2,965
- functional 인스턴스: 2,973
- Aff-Grasp train 샘플은 데이터 균형을 위해 2회 반복
- ignore가 있는 샘플은 YOLO에서 전체 제외

`contain`을 YOLO 변환본에서만 암묵적 배경으로 사용한
`data/processed/yolo_public_contain_background`:

- train: 2,836
- validation: 590
- test: 627
- grasp 인스턴스: 4,096
- functional 인스턴스: 2,973
- contain 예외 적용: 1,183개 원본 레코드
- 통합 semantic PNG의 `contain=ignore`는 그대로 보존

### 4.4 공개 데이터 사전학습과 평가

기본 모델: YOLO11n-seg, 입력 640, seed 42, AdamW, 80 epoch.

엄격 변환본에서 학습한 기존 D 모델의 고정 test 494장 mask 결과:

| 지표 | 전체 | grasp_region | functional_region |
|---|---:|---:|---:|
| mAP50 | 0.7600 | 0.6955 | 0.8245 |
| mAP50-95 | 0.4968 | 0.3936 | 0.6000 |
| Precision | 0.8416 | 0.7979 | 0.8854 |
| Recall | 0.7116 | 0.6651 | 0.7582 |

이 모델은 일반 test 지표는 초기 모델로 무난했지만, contain 정책 감사 117장 모두에서
컵·머그·냄비의 contain 영역을 `functional_region`으로 예측했다. 객체별 평균
오검출 덮임 비율은 약 96.6~99.0%였다. 이 수치는 formal test와 섞지 않고 별도 정책
감사 결과로 보관했다.

contain-background 변환본으로 다시 학습한 D 모델:

- 기존 고정 test 494장 mask mAP50: 0.7402
- 기존 고정 test 494장 mask mAP50-95: 0.4706
- 확장 test 627장 mask mAP50: 0.7599
- 정책 감사 117장에서 functional-on-contain: 0장

기존 494장 성능은 소폭 낮아졌지만 컵 몸통을 functional로 잡는 핵심 정책 오류가
사라졌으므로, 현재 자체 데이터 미세조정 실험 D는 이 checkpoint에서 시작하도록
설정했다.

checkpoint:

```text
outputs/training/public_pretrain_d_contain_background_seed42/weights/best.pt
```

관련 결과:

- `outputs/evaluation/public_pretrain_d_seed42_test`
- `outputs/evaluation/public_pretrain_d_contain_background_seed42_original_test`
- `outputs/evaluation/public_pretrain_d_contain_background_seed42_expanded_test`
- `docs/contain_background_experiment.md`

---

## 5. 지금까지 완료한 자체 데이터 작업

### 5.1 촬영 구성

원본은 `data/raw/custom/images`에 있고 수정하지 않았다.

| 객체 | 장수 | 분할 |
|---|---:|---|
| mug_01 | 43 | train |
| mug_02 | 47 | train |
| mug_03 | 50 | validation |
| 배경 음성 | 20 | train |
| 합계 | 160 | train 110 / validation 50 |

촬영 구간과 물체 ID는 `configs/datasets/custom_captures.yaml`에 기록돼 있다.

### 5.2 반자동 라벨링과 사람 검수

Grounding DINO와 SAM2로 손잡이/몸통 후보를 만든 뒤 사람이 수정했다. 후보는 정답으로
바로 사용하지 않았고 전체 140장을 수동 검수했다.

주요 파일:

| 파일 | 역할 |
|---|---|
| `src/labeling/custom_pseudolabels.py` | Grounding DINO+SAM2 후보 생성 지원 |
| `scripts/generate_custom_pseudolabels.py` | 서버 pseudo-label 생성 CLI |
| `src/labeling/custom_review.py` | 검수 workspace, mask 검사와 overlay |
| `scripts/review_custom_masks.py` | OpenCV 수동 수정 GUI |
| `scripts/prepare_full_custom_mask_review.py` | 전체 140장 수정 workspace 생성 |
| `전체_140장_마스크_수정_실행.bat` | Windows GUI 실행 편의 파일 |
| `src/labeling/custom_approval.py` | 사람 확정본을 승인 데이터로 승격 |
| `scripts/promote_custom_labels.py` | 승인본 생성 CLI |

최종 수동 검수 workspace:

```text
outputs/custom_mask_review/full_140_round2
```

140개 레코드는 모두 `review_status=human_corrected` 상태였다.

### 5.3 승인 데이터

승인본:

```text
data/interim/custom_approved_v1
```

내용:

- 객체 사진 140장
- 검증된 배경 음성 20장
- 총 160장
- 모든 객체 레코드 `annotation_status=human_verified`
- 배경 레코드 `annotation_status=verified_empty`
- 자체 데이터에서 `functional_region` 생성 안 함
- 승인 과정에서 64픽셀 미만 고립 조각만 정리
  - handle 558픽셀
  - body 346픽셀
- 원본과 수동 검수 workspace는 덮어쓰지 않음

### 5.4 자체 YOLO 데이터

최종 경로:

```text
data/processed/yolo_custom_approved_v1
```

최종 수량:

- train 이미지/라벨: 110
- validation 이미지/라벨: 50
- test: 0
- grasp polygon 인스턴스: 266
- 배경 빈 라벨: 20
- 전체 160장 모두 export

YOLO polygon은 작은 구멍이나 고립 조각 때문에 전체 컴포넌트를 거부할 수 있어서,
원본 승인 mask를 바꾸지 않고 YOLO 변환본에서만 다음 정리를 적용했다.

```text
max_secondary_contour_area = 512
max_hole_contour_area = 128
source_masks_modified = false
```

82개 컴포넌트에 작은 위상 정리가 적용됐고 모든 변경 픽셀과 원본 대비 IoU는
`export_manifest.jsonl`에 기록돼 있다.

표현하지 못해 제외된 컴포넌트는 딱 1개다.

```text
source_id: mug_02/WIN_20260814_16_50_22_Pro
component_id: 2 (body)
reason: component_not_single_hole_free_polygon
```

이 장면은 앞쪽 손잡이가 몸통을 가려 몸통 mask에 큰 구멍이 생긴 경우다. 큰 구멍을
배경으로 몰래 채우지 않았고, 몸통 컴포넌트만 제외했다. 같은 이미지의 손잡이와 이미지
자체는 학습에 포함된다.

이전 변환 순서가 있던 생성본은 삭제하지 않고 다음에 보관했다.

```text
data/processed/yolo_custom_approved_v1_before_orderfix
```

정식 경로는 반드시 `data/processed/yolo_custom_approved_v1`을 사용한다.

---

## 6. 테스트와 검증 상태

로컬과 서버 모두 다음 명령에서 34개 테스트를 통과했다.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

자체 데이터 검증:

```bash
python scripts/validate_dataset.py \
  data/interim/custom_approved_v1/manifest.jsonl

python scripts/validate_yolo_dataset.py \
  data/processed/yolo_custom_approved_v1
```

서버에서 확인한 결과:

- 압축 SHA-256 통과
- 최종 bundle 내부 394개 파일 통과
- 단위 테스트 34개 통과
- YOLO train 110 / validation 50 / test 0 통과
- class 0 polygon 266개 확인
- A 설정 서버 점검과 dry-run 통과
- D 설정 서버 점검과 dry-run 통과
- CUDA 사용 가능 확인

서버 완료 표시:

```text
/DATA/<계정>/hjh_vision_hand_storage/incoming/custom_final_ready.status
내용: CUSTOM_FINETUNE_FINAL_READY
```

최종 서버 묶음:

```text
artifacts/vision_hand_custom_finetune.tar.gz
정확한 SHA-256은 같은 폴더의
`vision_hand_custom_finetune.tar.gz.sha256`을 원본으로 사용한다.
```

서버 업로드 위치:

```text
/DATA/<계정>/hjh_vision_hand_storage/incoming/vision_hand_custom_finetune.tar.gz
```

---

## 7. 자체 미세조정 실험 설정

### 실험 A

```text
configs/training/custom_finetune_a_coco.yaml
```

- 초기값: `models/pretrained/yolo11n-seg.pt`
- 의미: COCO pretrained → 자체 데이터
- 결과 이름: `custom_finetune_a_coco_seed42`

### 실험 D

```text
configs/training/custom_finetune_d_public.yaml
```

- 초기값:
  `outputs/training/public_pretrain_d_contain_background_seed42/weights/best.pt`
- 의미: Aff-Grasp+UMD 사전학습 → 자체 데이터
- 결과 이름: `custom_finetune_d_public_seed42`

두 실험에서 같은 값:

- 데이터: `data/processed/yolo_custom_approved_v1/dataset.yaml`
- epochs: 100
- imgsz: 640
- seed: 42
- optimizer: AdamW
- lr0: 0.0003
- patience: 25
- device: 0
- batch: -1
- conservative augmentation
- mosaic/mixup/cutmix/copy-paste/erasing: 0

A와 D는 초기 가중치만 다르게 유지해야 한다. 비교 중 한쪽의 split, seed, 입력 크기,
증강이나 평가 코드를 임의로 바꾸면 안 된다.

주의: 현재 자체 데이터에는 positive `functional_region`이 없다. 자체 validation은
머그의 `grasp_region` 성능만 검증한다. D가 공개 데이터에서 배운
`functional_region`을 얼마나 보존하는지는 공개 고정 평가나 향후 별도 기능 영역
데이터로 확인해야 한다. 자체 validation 결과만으로 functional 성능이 좋다고 주장하면
안 된다.

---

## 8. Claude Code가 처음 해야 할 확인

### 로컬에서 시작

PowerShell:

```powershell
cd C:\Users\han02\Documents\sail\vision_hand
claude
```

Claude Code가 열린 뒤 먼저 다음을 확인한다.

```powershell
Get-Content AGENTS.md
Get-Content PROJECT_CONTEXT.md
Get-Content CLAUDE_CODE_HANDOFF.md
python -m unittest discover -s tests -p "test_*.py"
python scripts/validate_yolo_dataset.py data/processed/yolo_custom_approved_v1
```

### 서버 상태 확인

```bash
ssh <학습서버>          # 접속 정보는 별도 전달
tmux list-sessions
tmux list-windows -t 5
tmux attach -t 5
```

tmux 안에서:

```bash
cd ~/hjh_vision_hand
conda activate /DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_vision_hand_train
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
df -h /DATA
cat /DATA/<계정>/hjh_vision_hand_storage/incoming/custom_final_ready.status
```

위 명령은 읽기 전용이다. 비밀번호를 파일에 쓰지 않는다.

---

## 9. 바로 다음 단계: A/D smoke test

전체 학습 전에 각 설정으로 1 epoch, 데이터 5% smoke test를 수행한다. 반드시 tmux와
지정 conda 환경 안에서 실행한다.

```bash
cd ~/hjh_vision_hand
conda activate /DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_vision_hand_train

python scripts/check_training_server.py \
  --config configs/training/custom_finetune_d_public.yaml

python scripts/train_seg.py \
  --config configs/training/custom_finetune_d_public.yaml \
  --smoke-test

python scripts/check_training_server.py \
  --config configs/training/custom_finetune_a_coco.yaml

python scripts/train_seg.py \
  --config configs/training/custom_finetune_a_coco.yaml \
  --smoke-test
```

smoke test에서 확인할 것:

1. CUDA GPU가 실제로 사용되는가.
2. train/validation 이미지가 정상 로드되는가.
3. segmentation loss가 NaN/Inf가 아닌가.
4. 빈 배경 라벨 때문에 로더가 실패하지 않는가.
5. 결과 폴더에 `launch_metadata.json`, `run_status.json`, `weights`, 그래프가 생기는가.
6. `run_status.json`이 `completed`인가.

smoke test 수치를 최종 성능으로 보고하지 않는다. 같은 smoke 폴더가 이미 있으면
삭제하거나 `exist_ok=true`로 바꾸지 말고, 먼저 기존 실행 내용을 확인한 뒤 새 실행
이름을 정한다.

---

## 10. smoke 통과 후 전체 학습

권장 순서는 최종 후보인 D를 먼저 학습하고, 완료 후 비교 기준 A를 학습하는 것이다.
두 설정 모두 `device: 0`이므로 동시에 실행하지 않는다. 병렬 실행을 위해 device를
바꾸려면 실험 조건과 서버 점유 상태를 먼저 사용자에게 설명하고 승인받는다.

### D 전체 학습

tmux에서 새 창을 만들거나 `custom_setup` 창을 사용한다.

```bash
cd ~/hjh_vision_hand
conda activate /DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_vision_hand_train
set -o pipefail
python scripts/train_seg.py \
  --config configs/training/custom_finetune_d_public.yaml
```

학습을 시작한 뒤 초반 2~3 epoch까지만 확인한다.

- GPU 사용 정상
- loss 유한값
- validation 정상
- 저장 경로가 `/DATA/.../outputs/training` 아래인지 확인

그다음 `Ctrl+B`, `D`로 tmux에서 분리한다. 프로세스를 종료하지 않는다. 사용자가
“학습 완료”라고 알릴 때까지 반복 폴링하거나 계속 화면을 읽지 않는다.

### D 완료 후 A 전체 학습

사용자가 D 완료를 알린 뒤 결과 파일을 확인하고, 동일한 방식으로 A를 시작한다.

```bash
python scripts/train_seg.py \
  --config configs/training/custom_finetune_a_coco.yaml
```

역시 초반 2~3 epoch만 확인하고 tmux에서 분리한다.

---

## 11. 사용자가 학습 완료를 알렸을 때

각 실행에서 다음을 확인한다.

```bash
cd ~/hjh_vision_hand

cat outputs/training/custom_finetune_d_public_seed42/run_status.json
ls -lh outputs/training/custom_finetune_d_public_seed42/weights
tail -n 5 outputs/training/custom_finetune_d_public_seed42/results.csv

cat outputs/training/custom_finetune_a_coco_seed42/run_status.json
ls -lh outputs/training/custom_finetune_a_coco_seed42/weights
tail -n 5 outputs/training/custom_finetune_a_coco_seed42/results.csv
```

확인 항목:

- `best.pt`, `last.pt` 존재
- `run_status.json=completed`
- 실제 종료 epoch와 early stopping 여부
- best epoch의 validation mask mAP50, mAP50-95, Precision, Recall
- 실행 metadata의 데이터 hash, model hash, seed
- A와 D가 정말 같은 데이터와 설정을 썼는지
- 실패 실행도 숨기지 말고 기록

현재 custom test는 0장이므로 validation 결과만 비교한다. validation을 test라고 부르지
않고, `mug_03`에 대한 validation 성능이라고 명시한다.

학습 후 우선 작업:

1. A와 D validation 표 작성
2. 같은 validation 이미지에 A/D 예측 overlay 저장
3. 손잡이 누락, 몸통/손잡이 혼동, 배경 오검출 사례 분류
4. D가 A보다 실제로 나은지 수치와 육안 결과를 함께 판단
5. 선택한 `best.pt`로 로컬 외부 카메라 실시간 시험

---

## 12. `mug_04`를 추가한 뒤에만 최종 test

`mug_04`는 아직 촬영하지 않았다. 따라서 지금 test 수치나 가짜 test 이미지를 만들면
안 된다.

사용자가 네 번째 실제 컵을 촬영하면:

1. 실제 파일명과 촬영 시각 범위를 확인한다.
2. `configs/datasets/custom_captures.yaml`에 `object_id=mug_04`, `split=test`로 추가한다.
3. 기존 `mug_01/02/03` 분할은 바꾸지 않는다.
4. 원본을 보존하고 별도 v2 또는 test 전용 검수 workspace를 만든다.
5. Grounding DINO+SAM2는 초기 후보만 만들고 사람이 전부 검수한다.
6. 승인본을 `custom_approved_v2`처럼 새 버전으로 만든다.
7. 현재 v1 폴더를 덮어쓰지 않는다.
8. A/D의 고정 `best.pt`에 대해 같은 test로 평가한다.

중요: 현재 `promote_full_review()`는 기존 전체 object source와 review workspace ID가
정확히 같아야 한다. `mug_04`를 `data/interim/custom.jsonl`에 추가한 뒤 기존 140장
workspace에 그대로 재실행하면 ID 불일치로 실패하는 것이 정상이다. Claude Code는
기존 v1을 억지로 수정하지 말고 test 전용 승인 경로나 v2 승격 절차를 구현해야 한다.

test가 준비된 뒤 평가 명령의 형태:

```bash
python scripts/evaluate_test_seg.py \
  --model outputs/training/custom_finetune_d_public_seed42/weights/best.pt \
  --data data/processed/<새_custom_v2>/dataset.yaml \
  --output-dir outputs/evaluation/custom_finetune_d_mug04_test \
  --device 0 \
  --imgsz 640 \
  --expected-test-images <실제_mug04_test_장수>
```

A도 같은 test와 평가 코드로 실행한다. best checkpoint는 test 결과를 보고 다시 고르지
않는다.

---

## 13. 학습 다음의 전체 개발 순서

### 13.1 단기: 자체 모델 검증

1. A/D validation 비교
2. `mug_04` unseen-instance test 구축
3. test 정량 평가와 overlay 육안 검수
4. 배경 오검출, 손잡이 누락, 몸통 분할 실패 사례 정리
5. 필요하면 더 다양한 실제 물체와 배경 데이터를 추가

현재 자체 데이터는 머그 3개뿐이므로 모든 물체에 일반화된 모델이라고 주장하면 안 된다.

### 13.2 실시간 추론

이전에 외부 카메라로 직접 시험한 영상과 진단 결과는 있다.

- `outputs/camera_diagnostics`
- `outputs/realtime_test`

초기 공개 모델은 실제 광각 카메라에서 책상·벽 등 큰 배경 영역을 grasp/functional로
오검출했다. confidence threshold만 올려서는 충분히 해결되지 않았다. 그래서 자체
머그와 배경 음성 데이터를 추가했다.

현재 저장소에는 재현 가능한 정식 실시간 추론 스크립트가 없다. 학습 완료 후
`scripts/run_realtime_seg.py`와 `src/perception/` 모듈을 새로 구현하는 것이 좋다.

필수 기능:

- 카메라 장치 번호와 해상도 CLI/YAML 분리
- 모델 경로 CLI 인자
- confidence와 mask threshold 설정화
- 원본/overlay/녹화 저장 옵션
- 프레임 처리 FPS 표시
- 카메라 끊김 시 안전 종료
- 물체 클래스 이름 없이 affordance mask만 사용
- 왜곡이 심하면 카메라 calibration 후 undistortion 옵션

### 13.3 파지 후보 선택

모델 mask에서 centroid만 쓰지 않는다.

1. 같은 클래스의 분리 컴포넌트를 후보로 유지
2. distance transform으로 경계에서 안전한 내부점 계산
3. 손과의 거리, 접근 방향, 후보 폭, 기능 영역 안전거리 계산
4. confidence가 낮거나 후보 점수 차이가 작으면 `ALIGN`
5. 후보 폭에 따라 `PRECISION`, `WRAP`, `POWER` 결정

가중치와 폭 기준은 YAML로 분리하고 validation/실측 없이 확정하지 않는다.

### 13.4 ArUco와 실제 손

순서:

1. 카메라와 모델 독립 시험
2. ArUco 위치·방향 추정 독립 시험
3. Arduino serial mock test
4. Brunel Hand 세 프리셋 독립 시험
5. timeout과 `STOP` 경로 확인
6. 마지막에 비전과 손 제어 통합

사용자가 장치 연결과 전원 상태를 확인하기 전에는 실제 모터를 구동하지 않는다.

---

## 14. 아직 사용자에게 받아야 하는 실측 정보

다음 값은 추측하면 안 된다.

- Jetson 정확한 모델, JetPack, Python/CUDA 환경
- 외부 카메라 모델, 장치 번호, 해상도, FPS, 고정 위치
- 카메라 calibration 결과 또는 checkerboard 촬영본
- Arduino serial port와 baud rate
- Brunel Hand 액추에이터 수와 핀 매핑
- ArUco 마커 ID와 실제 크기
- 손의 실측 최대 개구 폭
- `PRECISION`, `WRAP`, `POWER` 실제 액추에이터 목표값
- `T_precision`, `T_wrap`, `R_precision`

값이 없으면 설정 schema, mock, dry-run까지 작성할 수 있지만 실제 장치를 구동하거나
실거리 수치를 확정하면 안 된다.

---

## 15. 자주 생길 수 있는 실수

1. `mug_03` validation을 test라고 부르지 않는다.
2. `mug_04` 촬영 전 test 성능을 만들지 않는다.
3. UMD `contain` 통합 PNG를 functional로 바꾸지 않는다.
4. contain-background 실험을 원본 라벨 정책 변경으로 잘못 설명하지 않는다.
5. 자체 데이터에 functional positive가 없다는 점을 보고서에서 숨기지 않는다.
6. 빈 배경 라벨 파일 20개를 데이터 오류로 삭제하지 않는다.
7. 손잡이와 몸통 instance를 합치지 않는다.
8. `data/processed/yolo_custom_approved_v1_before_orderfix`를 정식 데이터로 쓰지 않는다.
9. 현재 폴더에 `.git`이 없으므로 git 명령으로 사용자 파일을 복원하려 하지 않는다.
10. 서버에서 `sudo`, apt update, CUDA/driver 업데이트를 하지 않는다.
11. 학습 중 SSH가 끊겨도 tmux 프로세스를 새로 중복 실행하기 전에 기존 창과 GPU를 확인한다.
12. 전체 학습을 계속 감시하지 않는다. 초반 정상 동작만 확인하고 사용자 완료 알림을 기다린다.

---

## 16. Claude Code에 처음 보낼 권장 프롬프트

Claude Code를 저장소 루트에서 실행한 뒤 다음을 그대로 전달하면 된다.

```text
저장소 루트의 CLAUDE.md, AGENTS.md, PROJECT_CONTEXT.md,
CLAUDE_CODE_HANDOFF.md를 처음부터 끝까지 읽어.

라벨 정책과 객체 ID 기반 분할을 바꾸지 말고, 원본 데이터와 승인 mask를 덮어쓰지 마.
코드 주석, docstring, 오류 메시지, 문서는 한국어로 작성해.

먼저 로컬 파일과 테스트 상태를 읽기 전용으로 확인하고,
서버에서는 기존 tmux 세션 5와
/DATA/<계정>/hjh_vision_hand_storage/conda_envs/hjh_vision_hand_train
환경을 사용해. sudo나 서버 업데이트는 절대 하지 마.

다음 작업만 진행해:
1. custom_finetune_d_public과 custom_finetune_a_coco 설정의
   1 epoch smoke test를 각각 실행한다.
2. CUDA 사용, 데이터 로딩, loss NaN 여부, 결과 저장과 run_status를 확인한다.
3. smoke 결과를 나에게 보고하고 전체 학습은 아직 시작하지 않는다.

비밀번호를 파일이나 로그에 저장하지 마.
```

smoke 결과를 확인한 뒤 전체 D 학습을 시킬 때는 다음 프롬프트를 쓴다.

```text
smoke test가 모두 정상이라면 실험 D 전체 미세조정을 tmux에서 시작해.
초반 2~3 epoch가 정상 동작하는지만 확인하고 tmux에서 분리해.
학습을 계속 폴링하지 말고 종료하지도 마. 내가 완료됐다고 말하면 그때 결과를 확인해.
실험 A는 D가 완료된 뒤 내가 지시하면 시작해.
```

---

## 17. 핵심 파일 빠른 목록

### 반드시 먼저 읽기

- `CLAUDE.md`
- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `CLAUDE_CODE_HANDOFF.md`
- `docs/data_pipeline.md`
- `docs/contain_background_experiment.md`
- `docs/server_training.md`

### 현재 학습 입력

- `data/interim/custom_approved_v1/manifest.jsonl`
- `data/processed/yolo_custom_approved_v1/dataset.yaml`
- `data/processed/yolo_custom_approved_v1/dataset_version.json`
- `data/processed/yolo_custom_approved_v1/export_manifest.jsonl`

### 현재 학습 설정

- `configs/training/custom_finetune_a_coco.yaml`
- `configs/training/custom_finetune_d_public.yaml`

### 실행/검증

- `scripts/train_seg.py`
- `scripts/check_training_server.py`
- `scripts/validate_dataset.py`
- `scripts/validate_yolo_dataset.py`
- `scripts/evaluate_test_seg.py`
- `scripts/create_server_bundle.py`
- `scripts/verify_server_bundle.py`

### 승인 mask와 시각 확인

- `data/interim/custom_approved_v1/instance_masks`
- `data/interim/custom_approved_v1/semantic_masks`
- `data/interim/custom_approved_v1/overlays`
- `outputs/custom_mask_review/full_140_round2`
