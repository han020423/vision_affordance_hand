# UMD contain 배경 처리 비교 실험

## 목적

기존 YOLO 데이터셋은 `ignore=255` 픽셀이 하나라도 있으면 샘플 전체를 제외한다.
그 결과 컵·머그·국자·냄비처럼 `contain`과 올바른 `grasp_region`이 함께 있는
샘플도 학습에서 빠져, 컵 몸통 파지 누락과 입구의 `functional_region` 오검출이
발생할 가능성이 커졌다.

이 실험은 통합 라벨 정책을 바꾸지 않고 다음 차이만 검증한다.

- 통합 semantic PNG: 기존과 동일하게 `contain=ignore(255)`를 보존한다.
- 기본 YOLO 변환본: ignore 픽셀이 있으면 샘플 전체를 제외한다.
- 실험 YOLO 변환본: 사유가 오직 `umd_contain_policy`인 UMD 샘플만 허용하고,
  기존 `grasp_region` 폴리곤을 학습한다. contain 폴리곤은 기록하지 않아 YOLO가
  해당 영역을 암묵적 배경으로 처리한다.
- 알 수 없는 라벨, 라벨 충돌 등 다른 원인의 ignore는 계속 제외한다.

## 데이터 수량

실제 manifest와 폴리곤 변환 결과를 집계한 수량이다.

| 분할 | 기존 | 실험 | 증가 |
|---|---:|---:|---:|
| Train | 2,134 | 2,836 | 702 |
| Validation | 470 | 590 | 120 |
| Test | 494 | 627 | 133 |
| 합계 | 3,098 | 4,053 | 955 |

실험 대상 contain 샘플은 1,183장이지만, 228장은 YOLO 단일 폴리곤으로 표현할 수
있는 컴포넌트가 없어 최종 데이터에 추가되지 않았다. 이 제외 사유는
`export_manifest.jsonl`에 남는다.

## 재생성 명령

```bash
python scripts/export_yolo_public.py \
  --output data/processed/yolo_public_contain_background \
  --ignore-export-policy contain_as_background_keep_grasp

python scripts/validate_yolo_dataset.py \
  data/processed/yolo_public_contain_background
```

## 학습 명령

서버의 tmux와 지정 conda 환경 안에서 실행한다.

```bash
python scripts/train_seg.py \
  --config configs/training/public_pretrain_contain_background.yaml
```

기존 기준 모델과 데이터 정책의 영향만 비교하기 위해 COCO 사전학습 가중치,
seed, 입력 크기, optimizer, 증강과 epoch 설정을 동일하게 유지한다.

## 평가 원칙

학습 종료 후 다음 세 평가를 분리해 기록한다.

1. 기존 도구 test 494장: 기존 범주 성능이 나빠지지 않았는지 비교한다.
2. 확장 test 627장: 새로 포함된 contain 계열의 grasp 성능을 평가한다.
3. 기존 정책 감사 117장: 컵 입구의 functional 오검출과 몸통 grasp 누락을
   이전 모델과 같은 방법으로 비교한다.

이 방법은 진짜 픽셀 단위 ignore loss가 아니다. contain 픽셀을 배경으로 학습하는
비교 실험이므로, 개선이 부족하거나 기존 기능 영역 성능이 크게 하락하면
Ultralytics 데이터 로더와 segmentation loss에 유효 픽셀 마스크를 추가하는
별도 구현을 검토한다.
