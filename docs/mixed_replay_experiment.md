# 실험 E: 혼합(replay) 미세조정으로 망각 완화

## 배경과 진단 결과

실험 D(공개 사전학습 → 자체 머그 110장 미세조정)는 자체 validation(`mug_03`)에서는
정상 동작했지만, 실시간 카메라 시험에서 머그 외 물체를 거의 검출하지 못했다.
미세조정 D `best.pt`를 공개 고정 test 627장으로 평가한 진단 결과
(`outputs/evaluation/custom_finetune_d_seed42_public_test_after_finetune`):

| 지표 (mask) | 미세조정 전 사전학습 모델 | 미세조정 D |
|---|---:|---:|
| test 627장 mAP50 | 0.7599 | 0.1189 |
| functional_region 예측 없음 | - | 467/467장 (recall 0.0) |

자체 데이터에는 머그 `grasp_region`만 있고 functional positive가 없어서, 미세조정
과정에서 공개 데이터에서 배운 다른 범주와 `functional_region` 지식이 거의 완전히
망각(catastrophic forgetting)되었다.

## 실험 E 설계

- 초기 가중치: D와 동일한 `public_pretrain_d_contain_background_seed42/weights/best.pt`
- 데이터: 자체 승인 train(110장, 반복 10회) + 공개 train(2,836장)을 섞은
  `data/processed/yolo_mixed_replay_v1` (약 자체 28% 비중)
- validation: 자체 val 50장 + 공개 val 590장 병합. early stopping이 머그 성능과
  공개 데이터 성능을 함께 반영하도록 한다.
- 나머지 하이퍼파라미터(seed 42, lr0 0.0003, epochs 100, patience 25, 보수적 증강)는
  A/D와 동일하게 유지한다.

혼합 데이터셋은 `scripts/build_mixed_finetune_dataset.py`가 이미지 경로 목록(txt)
방식으로 생성하므로 원본 두 변환본의 파일은 복사·수정되지 않는다. 반복 배수와
수량, 원본 dataset_version 해시는 `mixed_manifest.json`에 기록된다.

주의: `scripts/check_training_server.py`의 내장 YOLO 데이터셋 검증은 export 형식
(images/labels 디렉터리 + export_manifest) 전용이라 txt 목록 방식인 혼합 데이터셋에는
"YOLO 데이터셋 검증에 실패했습니다"를 보고한다. 이는 구조 비호환이지 데이터 오류가
아니며, 대신 다음 두 가지로 무결성을 확인한다.

1. 원본 두 변환본 각각에 `scripts/validate_yolo_dataset.py` 통과 확인
2. build 스크립트 내장 검사(클래스 매핑, 라벨 존재, train/val 누수) 통과

## 기존 실험과의 관계

- A(COCO→자체), D(공개→자체)의 실행 결과와 비교 조건은 변경하지 않는다.
- E는 별도 추가 실험이며 `comparison_group: custom_finetune_forgetting_mitigation`으로
  구분한다.
- 라벨 정책(`grasp_region=0`, `functional_region=1`, UMD `contain=ignore 255`)과
  물체 ID 기반 분할(`mug_01/02=train`, `mug_03=val`, `mug_04=test 예약`)은 그대로다.

## 평가와 보고 원칙

학습 후 다음 세 평가를 분리해 보고한다. 자체 val과 공개 test 수치를 섞어 말하지 않는다.

1. 자체 val(`mug_03` 50장): A/D/E 비교 — 머그 성능이 D 대비 크게 나빠지지 않는지
2. 공개 test 627장: 사전학습 모델/미세조정 D/E 비교 — 망각 회복 정도
3. container 정책 감사 117장: functional-on-contain 0장 유지 확인

`mug_04`가 없으므로 자체 데이터에 대한 test 성능은 여전히 보고하지 않는다.
