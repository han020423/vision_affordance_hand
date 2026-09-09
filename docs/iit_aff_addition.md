# IIT-AFF 데이터셋 추가 (사용자 승인 2026-08-19)

## 목적과 승인

일반화 성능 향상을 위해 IIT-AFF(Nguyen et al. 2017, 8,835장)를 공개 사전학습
데이터에 추가했다. 기존 "Aff-Grasp와 UMD만 사용" 결정의 변경이며 사용자 승인을
받아 `AGENTS.md`에 반영했다. UMD가 회전판 위 단독 물체인 반면 IIT-AFF는 실제
어수선한 장면이라 배경 강건성과 형상 다양성을 보완한다.

- 출처: <https://sites.google.com/site/iitaffdataset/>
- 원본 위치: `data/raw/iit_aff/IIT_Affordances_2017` (수정 금지)
- 라벨 형식: 픽셀 단위 정수 행렬 txt (0=배경, 1~9=affordance)

## 라벨 매핑

| IIT-AFF 라벨 | 통합 저장값 | 3클래스 재매핑 |
|---|---|---|
| grasp(5) | grasp_region(1) | handle_grasp_region(0) |
| w-grasp(9) | grasp_region(1) | body_grasp_region(1) |
| cut(2), display(3), engine(4), hit(6), pound(7), support(8) | functional_region(2) | functional_region(2) |
| contain(1) | ignore(255) | YOLO 변환본에서만 암묵적 배경(contain-background 정책, 사유 `iit_contain_policy`) |
| 알 수 없는 값 | ignore(255) + human_review | 제외 |

`display`(모니터 화면)는 "현재 작업에서 피해야 하는 영역"이라는 프로젝트의
functional 정의에 따라 functional로 매핑했다. contain 정책은 UMD와 동일하게
유지한다 (컵·용기 몸통을 functional로 만들지 않음).

## 분할 정책

IIT-AFF는 장면 단위 데이터로 실제 물체 인스턴스 ID가 없다. 공식 train/val/test
분할은 물체 단위 격리를 보장하지 않으므로 사용하지 않고, Aff-Grasp 전례에 따라
**전량 `pretrain`(train 전용)**으로 기록했다. 따라서 공개 validation(1,179장)과
test(1,255장)는 IIT 추가 전(G 실험)과 완전히 동일해 IIT 효과를 공정하게 비교할
수 있다.

## 변환 결과 (2026-08-19)

- 발견 8,835 / 변환 8,512 / 제외 323 (빈 마스크 등, 사유는 manifest에 기록)
- 3클래스 공개 v2 (`data/processed/yolo_grasp_type_public_v2`):
  train 12,784 / val 1,179 / test 1,255
  인스턴스: handle 14,708 / body 3,074 / functional 13,748
  모호 다수결: UMD 5 + IIT 1

## 관련 실험

- `public_pretrain_grasp_type_s_v2_seed42`: v2 사전학습 (v1과 증강·모델 동일, 데이터만 IIT 추가)
- `custom_finetune_h_grasp_type_v2_seed42`: 실험 H — v2 사전학습 → 자체+공개 v2 replay 혼합 미세조정
- 비교 대상: 실험 G (IIT 없음, 같은 val/test)

## 구현 파일

- `src/datasets/iit_aff.py`: 파서와 매핑 (`prepare_iit_aff.py` CLI)
- `src/datasets/yolo_export.py`: contain 배경 정책의 IIT 사유 지원
- `src/datasets/yolo_export_grasp_type.py`: grasp(5) 대 w-grasp(9) 다수결 재매핑
- `tests/test_iit_aff.py`: 매핑·파서 단위 테스트
