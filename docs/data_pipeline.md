# 공개 Affordance 데이터 파이프라인

## 라벨 정책

모델 클래스는 `grasp_region=0`, `functional_region=1`이며 `ignore_index=255`를 유지한다. 조밀 PNG에는 배경값도 필요하므로, 손실 없이 저장하는 중간 의미 PNG는 다음 저장 인코딩을 사용한다.

| PNG 값 | 의미 | 모델 클래스 ID |
|---:|---|---:|
| 0 | 배경 | 해당 없음 |
| 1 | grasp_region | 0 |
| 2 | functional_region | 1 |
| 255 | ignore | 해당 없음 |

서로 떨어진 파지 영역과 기능 영역에는 인스턴스 PNG에서 각각 별도의 uint16 연결 성분 ID를 부여한다. 해당 모델 클래스 ID는 JSONL manifest에 저장한다. 이 방식으로 머그컵의 손잡이와 몸통처럼 같은 클래스인 후보가 하나로 합쳐지는 것을 막는다.

UMD의 `contain` 픽셀은 항상 `ignore=255`로 저장한다. 예상하지 못한 원본 값도 `ignore`로 처리하고 샘플 상태를 `human_review`로 기록한다. 임의로 한 클래스를 선택하지 않는다.

## 원본과 변환본 분리

- `data/raw/` 아래 원본 파일은 수정하지 않는 입력이다.
- Aff-Grasp `ego_train`은 공개 데이터 사전학습 입력이다.
- Aff-Grasp `Affordance_Evaluation_Dataset`은 외부 평가에만 사용한다.
- UMD ranked 라벨과 자동 생성 라벨은 기본 학습 입력으로 사용하지 않는다.
- 파생 마스크와 manifest는 `data/interim/` 아래에 저장한다.
- 모델별 변환본은 `data/processed/` 아래에 저장한다.

## 데이터 분할

`mug_01` 같은 UMD 물체 디렉터리 이름을 실제 `object_id`로 사용한다. 프레임을 임의로 나누지 않는다. 현재 고정 분할은 공식 fold 0을 test, 공식 fold 1을 validation, 나머지 실제 물체를 train으로 사용한다. `scripts/generate_umd_splits.py`는 선택한 fold가 서로 겹치지 않고 공식 ID가 다운로드한 물체 디렉터리와 정확히 일치하는지 검사한다. 로더도 하나의 물체가 여러 분할에 기록되면 거부한다.

공식 수동 정답 프레임만 선택 대상이다. 실제 물체별로 비례 배분하고 결정적인 시간 간격으로 선택해 기본 목표 4,000장을 정확히 맞춘다. 목표 수가 물체 105개보다 크므로 모든 물체를 유지한다. 자동 라벨 프레임과 선택되지 않은 수동 프레임은 제외 사유와 함께 manifest에 남긴다.

Aff-Grasp 학습 파일명은 신뢰할 수 있는 실제 물체 ID를 제공하지 않는다. 따라서 이 샘플은 `pretrain`, `object_id_status=not_provided_by_source`로 표시하고 validation이나 test로 임의 분할하지 않는다.

## 실행 명령

```powershell
python scripts/prepare_affgrasp.py --dry-run
python scripts/prepare_affgrasp.py

python scripts/generate_umd_splits.py

python scripts/prepare_umd.py --dry-run `
  --splits configs/datasets/umd_object_splits.yaml `
  --target-manual-frames 4000

python scripts/prepare_umd.py `
  --splits configs/datasets/umd_object_splits.yaml `
  --target-manual-frames 4000

python scripts/validate_dataset.py `
  data/interim/affgrasp.jsonl `
  data/interim/umd.jsonl

python scripts/render_manifest_overlays.py `
  data/interim/umd.jsonl `
  outputs/overlays/umd `
  --per-class 30

python scripts/export_yolo_public.py --dry-run --affgrasp-repeat 2
python scripts/export_yolo_public.py --affgrasp-repeat 2
python scripts/validate_yolo_dataset.py data/processed/yolo_public
```

## 자체 촬영 데이터 준비

자체 촬영 원본은 `data/raw/custom/images`에 그대로 보존한다. 촬영시각 구간과 실제
물체 ID의 대응은 `configs/datasets/custom_captures.yaml`에 명시하며, 프레임을 임의로
Train/Validation/Test에 나누지 않는다.

```powershell
python scripts/prepare_custom.py --dry-run
python scripts/prepare_custom.py
python scripts/validate_dataset.py data/interim/custom.jsonl
```

생성되는 `data/interim/custom_labeling`은 원본의 복사본과 라벨링 대상 경로를 담는다.
검증된 배경 음성 사진에는 전부 0인 semantic/instance PNG를 생성한다. 객체 사진은
정답 마스크가 없으므로 `pending_annotation`으로 유지하며, 사람이 검수하기 전까지
학습 데이터로 내보내지 않는다.

### Grounding DINO + SAM2 반자동 후보

서버에서는 기존 학습 환경을 직접 변경하지 않고 별도 conda 환경을 사용한다. 모델
가중치 캐시와 결과는 홈 공간이 아닌 추가 디스크에 둔다.

```bash
python scripts/generate_custom_pseudolabels.py \
  --sample-per-object 3 \
  --run-name sample_3_per_object \
  --device cuda:0 \
  --model-cache-dir /DATA/<계정>/hjh_vision_hand_storage/models/huggingface
```

결과는 `data/interim/custom_pseudolabels/<run-name>` 아래에 생성된다. 초록은 손잡이,
청록은 몸통 후보다. 두 부위는 모두 `grasp_region`이지만 instance ID는 분리한다.
머그에서 `functional_region`은 자동 생성하지 않는다. 이 폴더는 후보 전용이며,
`annotation_status=human_review_required`인 동안에는 학습 데이터로 내보내지 않는다.

검수할 때는 `priority_overlays` 폴더를 먼저 확인한다. 이 폴더에는 손잡이가
누락되었거나 너무 작게 분리된 것처럼 자동 경고가 붙은 이미지만 복사된다.
`검수_우선목록.txt`에는 이미지 ID와 경고 사유가 기록된다. 우선 목록 검수 후에는
경고가 없던 후보도 포함해 `overlays` 전체를 확인해야 한다. 초록 손잡이와 청록
몸통의 경계가 실제 파지 가능 영역과 다르면 승인하지 않고 수정 대상으로 남긴다.

### 자체 머그 후보 수동 수정

사용자가 잘못된 파일과 사유를 `configs/datasets/custom_human_review_20260818.yaml`에
기록한 뒤 수정용 복사본을 만든다. 원본 후보 폴더는 덮어쓰지 않는다.

```bash
python scripts/prepare_custom_mask_review.py
python scripts/review_custom_masks.py
```

수정 화면의 주요 단축키는 다음과 같다.

- `H`: 손잡이 선택, `B`: 몸통 선택, `E`: 배경으로 지우기
- 마우스 왼쪽 드래그: 선택 영역 칠하기, 오른쪽 드래그: 지우기
- `M`: 브러시/다각형 전환, 다각형 모드에서 `Enter`: 영역 채우기
- `[`/`]`: 브러시 크기 조절, `U`: 한 단계 되돌리기, `R`: 후보로 초기화
- `A`: 수정 승인 저장, `N`/`P`: 다음/이전 이미지, `Q`: 종료
- 손잡이가 카메라 반대편이라 실제로 안 보이면 손잡이 픽셀을 지우고 `V`로 비가시 표시
- 가림 때문에 정답을 판단할 수 없을 때만 `X`를 두 번 눌러 학습 제외

손잡이가 누락된 경우 `H`와 다각형 모드로 손잡이 외곽을 채운 뒤, 손잡이 구멍은
`E`와 다각형 모드로 다시 지운다. 몸통 윗부분이 누락된 경우 `B`로 해당 영역만
채운다. 저장 시 손잡이와 몸통은 각각 인스턴스 1과 2로 유지되지만 최종 의미
라벨은 둘 다 `grasp_region`이다. 수정한 항목도 전체 최종 오버레이를 다시 확인하기
전에는 학습 정답으로 승격하지 않는다.

최종 연락판에서 추가 오류가 보이면 현재 최종 마스크 140장 전체를 다시 수정 가능한
복사본으로 만들 수 있다. 정상 이미지는 `N`으로 넘기고 필요한 이미지만 수정해 `A`로
저장한다.

```bash
python scripts/prepare_full_custom_mask_review.py
python scripts/review_custom_masks.py \
  --workspace outputs/custom_mask_review/full_140_round2
```

Windows에서는 저장소 루트의 `전체_140장_마스크_수정_실행.bat`을 더블클릭해도 된다.
이 2차 수정 공간은 `final_140_v1`을 복사해 사용하므로 이전 수정본과 최종 확인본을
변경하지 않는다.

현재 고정 분할은 `mug_01`, `mug_02`를 Train, `mug_03`을 Validation으로 사용한다.
미촬영 `mug_04`는 최종 unseen-instance Test로 예약하며 실제 사진을 추가하기 전에는
Test 성능을 보고하지 않는다. 컵 몸통은 `functional_region`으로 지정하지 않고,
손잡이와 몸통의 파지 후보는 서로 다른 `grasp_region` 컴포넌트로 유지한다.

### 자체 데이터 최종 승인과 YOLO 변환

전체 140장을 사람이 저장 완료한 뒤에만 승인본으로 승격한다. 승인본은 수정 공간과
별도 폴더에 생성되어 수동 수정 마스크와 원본 이미지를 덮어쓰지 않는다. 기존에 사람이
확인한 배경 20장도 빈 정답으로 함께 포함한다.

```powershell
python scripts/promote_custom_labels.py
python scripts/validate_dataset.py data/interim/custom_approved_v1/manifest.jsonl

python scripts/export_yolo_public.py `
  --manifests data/interim/custom_approved_v1/manifest.jsonl `
  --output data/processed/yolo_custom_approved_v1 `
  --affgrasp-repeat 1 `
  --max-secondary-contour-area 512 `
  --max-hole-contour-area 128

python scripts/validate_yolo_dataset.py data/processed/yolo_custom_approved_v1
```

자체 승인 버전 `custom_approved_v1`은 물체 140장과 검증 배경 20장으로 구성된다.
분할은 Train 110장(`mug_01`, `mug_02`, 배경 20장), Validation 50장(`mug_03`)이다.
`mug_04`는 아직 촬영하지 않았으므로 Test는 비어 있으며 성능 수치를 만들지 않는다.

YOLO 폴리곤은 작은 점이나 구멍만으로도 변환에 실패할 수 있다. 위 두 임계값은 원본
승인 PNG를 수정하지 않고 YOLO 변환본에서만 작은 위상 잡음을 정리한다. 가장 큰 영역은
항상 보존하며, 제거·추가 픽셀 수와 원본 대비 폴리곤 IoU를 `export_manifest.jsonl`에
기록한다. 큰 구멍이나 복잡한 형상은 임의로 채우지 않고 해당 컴포넌트를 제외한다.

서버 미세조정 설정은 다음 두 개다. 데이터, 증강, seed는 같고 초기 가중치만 다르다.

- `configs/training/custom_finetune_a_coco.yaml`: COCO 사전학습에서 시작하는 실험 A
- `configs/training/custom_finetune_d_public.yaml`: Aff-Grasp+UMD 사전학습에서 시작하는 실험 D

## YOLO 폴리곤 내보내기 정책

기본 Ultralytics 폴리곤 형식에는 조밀한 `ignore=255` 정답을 표현할 방법이 없다. 따라서 ignore 픽셀이 하나라도 있는 UMD 샘플은 해당 픽셀을 몰래 배경으로 학습시키지 않고 샘플 전체를 제외한다. 제외 사유는 `export_manifest.jsonl`에 `ignore_pixels_not_supported_by_yolo_polygon`으로 기록한다.

구멍 없는 폴리곤 하나로 표현할 수 없는 연결 성분은 해당 성분만 제외하고 사유를 기록한다. 같은 샘플의 다른 유효 연결 성분은 계속 내보낼 수 있다. 내보낼 수 있는 성분이 하나도 없을 때만 샘플 전체를 제외한다. 원본 의미 PNG와 인스턴스 PNG는 수정하지 않는다.

생성 데이터셋은 `train`, `val`, `test` 디렉터리를 사용한다. Aff-Grasp의 `pretrain` 레코드는 `train`에만 매핑하고 UMD 편중을 줄이기 위해 두 번 반복한다. UMD validation과 test는 고정된 실제 물체 분할을 계속 사용하며, 학습 명령은 test를 사용하지 않는다.

다운로드한 UMD MAT 파일에는 공식 `gt_type` 필드가 있다. 값이 `manual`인 파일만 변환하고 `automatic` 파일은 제외로 기록한다. `--manual-frame-modulo`는 `gt_type`이 없는 다른 압축본을 위한 명시적 대체 옵션일 뿐이며 현재 배포본에는 사용하지 않는다.
