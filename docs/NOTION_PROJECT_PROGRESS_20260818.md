# 비전 기반 적응형 로봇손 파지 프로젝트 중간 진행 보고서

| 구분 | 내용 |
|---|---|
| 기준일 | 2026년 8월 18일 |
| 현재 단계 | 공개 데이터 사전학습 및 자체 머그 데이터 구축 완료, 자체 데이터 미세조정 준비 |
| 기본 모델 | YOLO11n-seg |
| 최종 목표 | Affordance Segmentation 결과를 Brunel Hand의 적응형 파지로 연결 |

## 1. 보고서 개요

본 보고서는 Aff-Grasp와 UMD 데이터 통합부터 공개 데이터 사전학습, 실제 카메라 점검, 자체 머그 데이터 구축까지의 작업 내용과 결과를 정리한 중간 보고서다. 학습 지표뿐 아니라 데이터 변환 과정에서 확인된 정책 문제, 자체 데이터 검수 내역, 현재 결과의 제한과 후속 실험 계획을 함께 기록했다.

현재 공개 데이터 통합 및 사전학습, 고정 test 평가, UMD `contain` 영역 오분류 개선, 외부 카메라 진단을 완료했다. 이후 머그컵 3개와 빈 배경을 촬영해 자체 데이터 160장을 구성했고, 객체 이미지 140장의 마스크를 전량 수동 검수했다. 서버 배치와 실험 A/D 설정 검증까지 마쳤으며 실제 자체 데이터 미세조정은 아직 시작하지 않았다.

| 구분 | 주요 내용 | 상태 |
|---|---|---|
| 공개 데이터 | Aff-Grasp·UMD 변환, 물체 ID 분할, YOLO 변환 | 완료 |
| 공개 모델 | 사전학습, test 494장 평가, 예측 마스크 검수 | 완료 |
| 정책 검증 | `contain=ignore`와 YOLO 형식 충돌 분석 및 개선 실험 | 완료 |
| 실제 카메라 | 외부 USB 카메라 추론, 배경 오검출 및 광각 문제 확인 | 완료 |
| 자체 데이터 | 머그 3개 140장, 빈 배경 20장 수집 | 완료 |
| 라벨 검수 | Grounding DINO·SAM2 초기 마스크 생성 및 140장 수동 수정 | 완료 |
| 학습 준비 | 자체 YOLO 변환, 서버 배치, A/D dry-run | 완료 |
| 자체 미세조정 | A/D smoke test 및 전체 학습 | 미수행 |
| 최종 시험 | `mug_04` 촬영, unseen-instance test | 미수행 |
| 시스템 통합 | 파지 후보 선택, ArUco, Brunel Hand 제어 | 미수행 |

---

## 2. 프로젝트 정의

### 프로젝트 목표

외부 고정 RGB 카메라 영상에서 물체의 잡을 수 있는 영역과 기능 영역을 분할하고,
로봇손이 접근하는 위치와 방향에 따라 가장 적절한 파지 후보를 선택한 다음 Brunel
Hand를 적절한 자세로 구동하는 시스템을 만드는 것이 목표다.

### 전체 시스템 흐름

```text
외부 고정 RGB 카메라
        ↓
YOLO11n-seg Affordance Segmentation
        ↓
grasp_region / functional_region mask
        +
손목 ArUco 마커 위치와 접근 방향
        ↓
복수 파지 후보 특징 및 접근 가능성 계산
        ↓
파지 후보 선택
        ↓
PRECISION / WRAP / POWER 결정
        ↓
Jetson → USB Serial → Arduino → Brunel Hand
```

### 연구 범위

- 외부 RGB 카메라를 사용한다.
- 로봇팔을 이용한 완전 자동 6-DoF grasp planning은 포함하지 않는다.
- 사용자가 로봇손을 물체 근처까지 이동한 뒤 시스템이 파지 위치와 손 모양을 결정하는
  반자율 시스템을 목표로 한다.
- 모든 임의 물체에 대한 open-world 일반화를 주장하지 않는다.
- 학습하지 않은 실제 물체 인스턴스와 제한된 미학습 범주에 대한 평가를 목표로 한다.

---

## 3. 통합 라벨 정책

### 최종 모델 클래스

```yaml
0: grasp_region
1: functional_region
ignore_index: 255
```

중간 semantic PNG는 모델 클래스 ID와 직접 같지 않고 다음 저장값을 사용한다.

| PNG 저장값 | 의미 | 모델 클래스 ID |
|---:|---|---:|
| 0 | background | 없음 |
| 1 | grasp_region | 0 |
| 2 | functional_region | 1 |
| 255 | ignore | 없음 |

### Aff-Grasp 매핑

| 원본 라벨 | 통합 라벨 |
|---|---|
| graspable | grasp_region |
| functional | functional_region |

### UMD 매핑

| 원본 라벨 | 통합 라벨 |
|---|---|
| grasp | grasp_region |
| wrap-grasp | grasp_region |
| cut | functional_region |
| scoop | functional_region |
| pound | functional_region |
| support | functional_region |
| contain | ignore |

### `contain`을 ignore로 둔 이유

UMD의 컵·머그·냄비 몸통에는 `contain` 의미가 부여된다. 이것을
`functional_region`으로 바꾸면 컵 몸통 전체가 피해야 할 영역이 되어 몸통 파지가
불가능해진다. 본 프로젝트에서는 머그 몸통도 유효한 `POWER` 파지 후보이므로
`contain`을 통합 라벨에서 `ignore=255`로 보존하기로 했다.

### 머그 손잡이와 몸통

- 손잡이와 몸통은 둘 다 `grasp_region`이다.
- 단, 서로 다른 파지 방법을 사용하므로 하나의 mask로 합치지 않는다.
- 손잡이는 `WRAP`, 몸통은 `POWER` 후보가 될 수 있다.
- instance mask에서는 손잡이와 몸통을 별도 ID로 유지한다.

---

## 4. 저장소 및 데이터 파이프라인 구성

원본, 통합 mask, 모델 입력 형식을 분리했다.

```text
data/raw       원본 데이터, 수정 금지
data/interim   통합 semantic/instance mask와 manifest
data/processed YOLO 등 모델별 변환본
outputs        학습·평가·오버레이·검수 결과
configs        데이터 및 학습 설정
scripts        실행용 CLI
src            파서, 변환, 라벨링 모듈
tests          단위 테스트
```

### manifest에 기록한 주요 정보

- `source_dataset`
- `source_id`
- `object_id`
- `original_labels`
- `mapped_labels`
- `split`
- `conversion_status`
- 이미지와 mask 경로
- component별 클래스와 픽셀 수
- 제외·ignore·검수 사유

### 중요한 데이터 처리 원칙

- 원본 파일은 수정하지 않는다.
- 변환은 재실행 가능하고 결정적으로 수행한다.
- 문서의 예상 이미지 수를 코드 상수로 사용하지 않고 실제 파일을 집계한다.
- 같은 실제 물체를 train과 validation/test에 동시에 넣지 않는다.
- UMD 회전판 연속 프레임을 임의 프레임 단위 random split하지 않는다.
- 알 수 없거나 충돌하는 픽셀을 억지로 한 클래스에 배정하지 않는다.

---

## 5. Aff-Grasp 및 UMD 공개 데이터 준비

### Aff-Grasp

Aff-Grasp는 `graspable`과 `functional` 구분이 비교적 직접적이어서 두 통합 클래스를
학습하는 데 사용했다. 학습 데이터만 공개 데이터 사전학습에 넣고 Affordance
Evaluation Dataset은 외부 평가용으로 유지했다.

Aff-Grasp에는 신뢰할 수 있는 실제 물체 ID가 충분히 제공되지 않아 별도의 validation과
test를 임의로 만들지 않고 `pretrain` 용도로만 사용했다.

### UMD

UMD는 다양한 도구 범주와 실제 물체 인스턴스가 있으나 회전판에서 촬영한 유사 프레임이
많다. 수동 GT 프레임만 사용하고 실제 물체 ID 기준으로 분할했다.

처리 방식:

1. 자동 생성 annotation을 기본 학습에서 제외
2. 수동 annotation 프레임만 선택
3. 실제 물체별로 프레임을 비례 배분
4. 시간 순서에서 결정적으로 서브샘플링
5. 공식 물체 ID fold를 사용해 train/validation/test 분리
6. 동일 물체 ID의 split 누수를 자동 검사

### 구현한 주요 도구

- `scripts/prepare_affgrasp.py`
- `scripts/generate_umd_splits.py`
- `scripts/prepare_umd.py`
- `scripts/validate_dataset.py`
- `scripts/render_manifest_overlays.py`
- `scripts/export_yolo_public.py`
- `scripts/validate_yolo_dataset.py`

---

## 6. 공개 데이터 YOLO 변환 결과

### 기본 엄격 변환본

경로:

```text
data/processed/yolo_public
```

| 항목 | 수량 |
|---|---:|
| 전체 원본 manifest 레코드 | 29,174 |
| Train 이미지 | 2,134 |
| Validation 이미지 | 470 |
| Test 이미지 | 494 |
| grasp_region 인스턴스 | 2,965 |
| functional_region 인스턴스 | 2,973 |

UMD 데이터가 Aff-Grasp보다 훨씬 많아 Aff-Grasp 학습 샘플을 두 번 반복해 균형을
일부 보완했다.

### YOLO 형식의 한계

Ultralytics의 일반 polygon segmentation 형식은 `ignore=255`를 픽셀 단위로 표현할 수
없다. 처음에는 ignore가 하나라도 있는 샘플을 전체 제외하는 보수적 정책을 사용했다.

또한 다음 component는 자동으로 억지 변환하지 않고 제외했다.

- 점이 3개보다 적어 polygon이 될 수 없는 mask
- 여러 고립 영역으로 나뉜 component
- 내부 구멍이 있어 단일 hole-free polygon으로 표현할 수 없는 component

제외 사유는 모두 `export_manifest.jsonl`에 기록했다.

---

## 7. 공개 데이터 사전학습 및 평가

### 학습 설정

- 모델: YOLO11n-seg
- 입력 크기: 640×640
- epoch: 80
- optimizer: AdamW
- 초기 learning rate: 0.001
- seed: 42
- AMP 사용
- validation 및 early stopping 사용
- mosaic, MixUp, CutMix, Copy-Paste는 사용하지 않음

### 고정 test 평가

`best.pt`를 고정한 뒤 UMD test 494장을 별도로 평가했다. validation 결과와 섞지 않고
별도 출력 폴더에 저장했다.

#### Mask 결과

| 지표 | 전체 | grasp_region | functional_region |
|---|---:|---:|---:|
| mAP50 | 0.7600 | 0.6955 | 0.8245 |
| mAP50-95 | 0.4968 | 0.3936 | 0.6000 |
| Precision | 0.8416 | 0.7979 | 0.8854 |
| Recall | 0.7116 | 0.6651 | 0.7582 |

### 초기 결과에 대한 판단

초기 공개 데이터 모델로는 전체 mask mAP50 약 0.76, mAP50-95 약 0.50 수준이었다.
기능 영역은 비교적 잘 잡았지만 grasp 영역 성능이 더 낮았다. 첫 기준 모델로는 사용할
수 있으나 실제 카메라 환경과 머그 손잡이/몸통 구분에는 추가 개선이 필요했다.

---

## 8. `contain/ignore` 정책 문제 분석 및 개선

### 발견한 문제

고정 test의 일반 지표만 보면 모델이 무난해 보였지만 컵·머그·냄비를 별도로 확인하자
심각한 문제가 보였다.

정책 감사 117장의 모든 이미지에서 모델이 원래 `contain=ignore`인 영역을
`functional_region`으로 예측했다.

| 객체 | 이미지 수 | functional 오검출 이미지 | 평균 덮임 비율 |
|---|---:|---:|---:|
| cup_03 | 35 | 35 | 약 96.6% |
| mug_20 | 47 | 47 | 약 99.0% |
| pot_02 | 35 | 35 | 약 97.5% |

즉, 일반 mAP는 나쁘지 않았지만 프로젝트 정책 관점에서는 컵 몸통이나 용기 영역을
거의 전부 기능 영역으로 취급하는 문제가 있었다.

### 원인 분석

통합 PNG의 `contain`은 올바르게 ignore로 유지했지만, YOLO polygon 형식에서 ignore를
표현할 수 없어 해당 샘플 전체를 학습에서 제외했다. 그 결과 모델은 컵·머그·냄비에서
`contain` 주변의 올바른 grasp 형태를 충분히 보지 못했다.

### 비교 실험

통합 PNG 정책은 바꾸지 않고 다음 제한적 YOLO 학습 예외를 만들었다.

- 사유가 정확히 `umd_contain_policy`인 UMD 샘플만 허용
- 기존 `grasp_region` polygon은 학습에 사용
- `contain` polygon은 기록하지 않음
- YOLO 학습에서 contain 영역은 암묵적 background가 됨
- 알 수 없는 라벨이나 충돌 등 다른 ignore 사유는 계속 제외

경로:

```text
data/processed/yolo_public_contain_background
```

| 분할 | 기존 | 비교 실험 | 증가 |
|---|---:|---:|---:|
| Train | 2,134 | 2,836 | 702 |
| Validation | 470 | 590 | 120 |
| Test | 494 | 627 | 133 |

### 비교 실험 결과

#### 기존 고정 test 494장

- mask mAP50: 0.7402
- mask mAP50-95: 0.4706

기존 엄격 모델의 0.7600/0.4968보다 일반 test 성능은 소폭 낮아졌다.

#### 확장 test 627장

- mask mAP50: 0.7599
- mask mAP50-95: 0.4883

#### 정책 감사 117장

- 기존 모델: 117장 모두 functional-on-contain 발생
- 비교 실험 모델: 117장 모두 0으로 감소

### 최종 판단

일반 test 수치는 약간 낮아졌지만 컵 몸통을 `functional_region`으로 보는 핵심 정책
오류가 사라졌다. 따라서 자체 데이터 미세조정 실험 D의 시작 checkpoint로
contain-background 모델을 선택했다.

중요한 표현:

> 원본 통합 라벨 정책을 바꾼 것이 아니라, ignore를 표현하지 못하는 YOLO 학습 형식의
> 한계를 보완하기 위한 별도 변환 실험이다.

---

## 9. 외부 카메라 실시간 추론 점검

공개 데이터 사전학습 모델을 로컬 외부 USB 카메라에 연결해 실시간으로 시험했다.

### 관찰한 문제

- 머그컵뿐 아니라 책상, 벽, 모니터 등 넓은 배경을 큰 mask로 예측
- 손잡이와 무관한 배경을 `grasp_region`으로 검출
- 화면 상단이나 책상 영역을 `functional_region`으로 검출
- confidence threshold를 올려도 큰 배경 mask가 충분히 사라지지 않음
- 카메라가 광각이라 공개 데이터와 실제 영상의 시점·왜곡 차이가 큼

### 판단

단순히 confidence를 높이는 것으로 해결하기 어렵고, 실제 카메라와 같은 배경·시점의
자체 데이터와 객체가 없는 negative 이미지가 필요하다고 판단했다.

### 남은 카메라 관련 개선

- 카메라 고정 위치와 작업 영역 확정
- calibration을 통한 광각 왜곡 보정 검토
- 학습과 추론에서 동일한 해상도·시점 유지
- 정식 실시간 추론 스크립트 구현
- 배경 false positive와 FPS 기록

진단 결과는 다음 경로에 있다.

```text
outputs/camera_diagnostics
outputs/realtime_test
```

---

## 10. 자체 머그 데이터 수집

### 촬영 목적

- 실제 사용할 외부 카메라의 영상 분포 반영
- 머그 손잡이와 몸통을 서로 다른 파지 후보로 학습
- 흰 촬영 부스와 실제 책상 배경을 모두 포함
- 배경만 있는 영상을 negative로 사용해 큰 배경 오검출 완화

### 촬영 수량과 분할

| 대상 | 장수 | 배경 | 분할 |
|---|---:|---|---|
| mug_01 | 43 | 흰 부스+책상 | Train |
| mug_02 | 47 | 흰 부스+책상 | Train |
| mug_03 | 50 | 흰 부스+책상 | Validation |
| 배경만 있는 사진 | 20 | 흰 부스+책상 | Train |
| 합계 | 160 |  | Train 110 / Validation 50 |

`mug_04`는 unseen-instance Test용으로 예약했으며 아직 촬영하지 않았다.

### 분할 기준

- `mug_01`, `mug_02`: train
- `mug_03`: validation
- `mug_04`: test 예정
- 같은 실제 컵의 프레임을 서로 다른 분할에 넣지 않음

촬영 구간과 실제 물체 ID는 `configs/datasets/custom_captures.yaml`에 기록했다.

---

## 11. Grounding DINO·SAM2 반자동 라벨링

### 사용 목적

Foundation Model을 최종 정답 생성기로 사용하지 않고, 사람이 수정할 초기 mask를 만드는
보조 도구로 사용했다.

### 처리 흐름

```text
원본 이미지
  ↓
Grounding DINO 텍스트 프롬프트
  ↓
손잡이/몸통 box
  ↓
SAM2 mask 생성
  ↓
품질 경고 및 overlay 생성
  ↓
사람이 전체 확인·수정
```

머그 프롬프트는 `mug handle`, `mug body`를 중심으로 사용했다.

### 색상 의미

- 초록색: handle 후보
- 청록색: body 후보

두 영역은 최종 의미 클래스는 같지만 서로 다른 instance로 유지했다.

### 자동 후보에서 자주 나타난 오류

- 손잡이 완전 누락
- 손잡이 영역이 너무 작음
- 몸통 일부를 손잡이로 잘못 표시
- 손잡이 mask가 몸통까지 과하게 확장
- 몸통 윗부분 또는 테두리 누락
- 손잡이가 보이지 않는 각도에서 억지 mask 생성

이 때문에 pseudo-label을 자동 승인하지 않고 전체 이미지를 수동으로 검수했다.

---

## 12. 객체 이미지 140장 수동 검수

### 검수 프로그램

OpenCV 기반 수동 mask 편집기를 만들었다.

주요 단축키:

- `H`: 손잡이 선택
- `B`: 몸통 선택
- `E`: 배경으로 지우기
- `M`: 브러시/다각형 모드 전환
- `Enter`: 다각형 채우기
- `[` / `]`: 브러시 크기 조절
- `U`: 되돌리기
- `R`: 초기 후보로 복원
- `A`: 수정 승인 저장
- `N` / `P`: 다음/이전 이미지
- `V`: 손잡이가 실제로 보이지 않는 이미지로 표시
- `Q`: 종료

### 진행 방식

1. 자동 후보 중 이상한 이미지와 사유를 먼저 목록화
2. 우선 검수 이미지를 수정
3. 우선 목록에 없던 이미지까지 포함해 전체 140장 확인
4. 최종 contact sheet와 개별 overlay 확인
5. 전체 140장을 다시 편집할 수 있는 별도 workspace 생성
6. 사용자가 모든 이미지를 수정 또는 확인한 뒤 승인 단계 진행

최종 검수 workspace:

```text
outputs/custom_mask_review/full_140_round2
```

전체 140개 레코드는 `human_corrected` 상태로 확인됐다.

---

## 13. 자체 승인 데이터 생성

사람이 검수한 결과를 후보 데이터와 분리해 새로운 승인 버전으로 승격했다.

```text
data/interim/custom_approved_v1
```

### 승인본 구성

- 사람이 검수한 객체 이미지: 140장
- 사람이 확인한 빈 배경 이미지: 20장
- 총 160장
- 객체 상태: `human_verified`
- 배경 상태: `verified_empty`
- 자체 머그에서 functional_region 생성 안 함
- 원본 이미지와 수동 검수 workspace는 보존

### 작은 brush artifact 정리

수동 편집 중 생길 수 있는 64픽셀 미만 고립 조각만 정리했다.

| 부위 | 제거한 픽셀 |
|---|---:|
| handle | 558 |
| body | 346 |

이 값은 모델 성능이 아니라 승인 과정에서 제거한 작은 mask artifact 수다.

---

## 14. 자체 데이터 YOLO 변환

최종 경로:

```text
data/processed/yolo_custom_approved_v1
```

### 최종 수량

| 항목 | 수량 |
|---|---:|
| 전체 이미지 | 160 |
| Train | 110 |
| Validation | 50 |
| Test | 0 |
| grasp polygon 인스턴스 | 266 |
| 빈 배경 라벨 | 20 |

### 작은 위상 잡음 처리

사람이 보기에는 정상인 mask도 내부의 몇 픽셀 구멍이나 떨어진 작은 조각 때문에 YOLO
단일 polygon 변환에 실패하는 문제가 있었다.

원본 승인 PNG는 수정하지 않고 YOLO 변환본에서만 다음 임계값을 적용했다.

```text
작은 분리 윤곽 최대 면적: 512
작은 내부 구멍 최대 면적: 128
원본 승인 mask 수정: false
```

82개 component에 이 정리가 적용됐다. 변경량과 원본 대비 polygon IoU는 모두
`export_manifest.jsonl`에 기록했다.

### 제외된 단 하나의 component

```text
mug_02/WIN_20260814_16_50_22_Pro
component_id=2, body
```

앞쪽의 손잡이가 몸통을 가려 body mask에 큰 구멍이 생긴 장면이다. 큰 구멍을 임의로
채우면 손잡이 영역까지 몸통 정답이 되므로 body component만 제외했다. 해당 이미지와
손잡이 component는 그대로 학습에 포함했다.

---

## 15. 서버 학습 환경 및 실행 검증

### 서버 사용 원칙

- tmux 필수
- conda 환경 필수
- `sudo` 사용 금지
- 서버 OS, CUDA, 드라이버 업데이트 금지
- 데이터와 학습 결과는 `/DATA` 또는 `/data2` 사용
- 홈에는 코드만 유지

비밀번호와 민감한 접속 정보는 노트나 저장소에 기록하지 않는다.

### 확인한 환경

- GPU: RTX 3090 24GB급 4장
- Python: 3.10.20
- PyTorch: 2.7.1+cu118
- Ultralytics: 8.3.163
- 학습 데이터와 결과 폴더는 `/DATA`로 연결
- 자체 미세조정 bundle SHA-256 확인 완료
- bundle 내부 파일 검증 완료
- 단위 테스트 34개 통과
- 자체 YOLO 데이터 검증 통과
- 실험 A/D dry-run 통과

### 서버용 재현성 장치

- 실행 전 데이터셋과 checkpoint hash 기록
- 설정 YAML hash 기록
- Python, PyTorch, CUDA, GPU, Ultralytics 버전 기록
- 실행별 폴더 분리
- 기존 결과 폴더 덮어쓰기 금지
- 성공·실패 여부를 `run_status.json`에 기록

---

## 16. 자체 데이터 미세조정 계획

### 실험 A: 자체 데이터 기준선

```text
COCO pretrained YOLO11n-seg
        ↓
자체 머그 Train 데이터
```

설정:

```text
configs/training/custom_finetune_a_coco.yaml
```

### 실험 D: 공개 Affordance 사전학습 활용

```text
Aff-Grasp + UMD contain-background 사전학습
        ↓
자체 머그 Train 데이터
```

설정:

```text
configs/training/custom_finetune_d_public.yaml
```

### 두 실험의 공통 조건

| 항목 | 값 |
|---|---|
| 데이터 | yolo_custom_approved_v1 |
| Epoch | 100 |
| 입력 크기 | 640 |
| Optimizer | AdamW |
| 초기 LR | 0.0003 |
| Seed | 42 |
| Patience | 25 |
| Batch | GPU 자동 결정 |
| Mosaic/MixUp/CutMix | 사용 안 함 |

두 실험은 초기 checkpoint만 다르고 나머지 데이터, split, 증강, seed와 평가 코드는
동일하게 유지한다.

### 현재 정확한 상태

설정 파일과 서버 dry-run은 완료했지만 실제 smoke test와 전체 학습은 시작하지 않았다.

다음 실행 순서:

1. D 1 epoch smoke test
2. A 1 epoch smoke test
3. smoke 결과 확인
4. D 전체 학습 시작
5. 초반 2~3 epoch만 정상 여부 확인 후 tmux 분리
6. D 완료 후 결과 확인
7. A 전체 학습 시작
8. A 완료 후 같은 validation으로 비교

---

## 17. 결과 해석의 제한

### 자체 validation은 머그 1개다

현재 validation은 실제 물체 `mug_03`의 50장이다. 프레임 수는 50장이지만 물리적
물체는 1개이므로 일반화 결론을 크게 주장하면 안 된다.

### 자체 데이터에는 functional positive가 없다

자체 머그 데이터에는 `grasp_region`과 background만 있다. 따라서 자체 validation으로
`functional_region` 성능을 평가할 수 없다.

- 실험 A는 자체 데이터만으로 functional을 새로 배울 수 없다.
- 실험 D는 공개 데이터에서 배운 functional 표현을 보존할 가능성이 있다.
- 이 차이는 공개 고정 평가 또는 기능 영역이 포함된 추가 자체 데이터로 확인해야 한다.

### test는 아직 없다

`mug_04`를 촬영하지 않았기 때문에 자체 test는 0장이다. 현재 나오는 자체 데이터 수치는
validation이며 최종 test 성능으로 보고하면 안 된다.

### 현재 데이터만으로 가능한 주장

- 학습하지 않은 머그 인스턴스 한 개에 대한 validation
- 제한된 머그 환경에서의 손잡이/몸통 grasp segmentation
- 공개 Affordance 사전학습의 자체 머그 미세조정 기여 비교

### 현재 데이터만으로 하면 안 되는 주장

- 모든 물체에 대한 open-world 일반화
- 모든 컵과 도구 범주에 대한 강한 일반화
- 자체 환경에서 functional_region이 충분히 검증됐다는 주장
- 네 번째 실제 물체 없이 unseen-instance test를 완료했다는 주장

---

## 18. 후속 작업 계획

### 단계 1. A/D smoke test 및 전체 학습

- CUDA와 데이터 로더 확인
- NaN/Inf loss 확인
- background 빈 라벨 처리 확인
- D와 A 순차 학습
- best epoch와 validation 지표 비교

### 단계 2. A/D 육안 비교

- 같은 `mug_03` validation 이미지에 A/D mask overlay 저장
- 손잡이 누락 사례
- 몸통과 손잡이 혼동 사례
- 배경 오검출 사례
- 작은 손잡이와 가림 각도 실패 사례
- 모델별 실패 패턴 표 작성

### 단계 3. `mug_04` 최종 test 구축

- 네 번째 실제 컵 촬영
- 기존 분할을 바꾸지 않고 test로만 배정
- Grounding DINO+SAM2 초기 후보 생성
- 모든 test mask 사람 검수
- 승인 데이터 v2 생성
- A/D의 고정 `best.pt` 평가
- 클래스별 mAP, Precision, Recall과 overlay 저장

### 단계 4. 자체 데이터 확장

머그 이외 범주가 필요하면 다음 순서로 확장한다.

1. 병 또는 원통형 용기
2. 망치 또는 고무망치
3. 가위
4. 프라이팬 또는 손잡이 용기
5. 주걱·국자

범주마다 가능한 한 실제 물체 4개를 준비해 train 2, validation 1, test 1로 분리한다.

### 단계 5. 정식 실시간 추론 프로그램

구현할 기능:

- 모델 경로 설정
- 카메라 번호와 해상도 설정
- confidence/mask threshold 설정
- FPS 표시
- overlay 및 영상 저장
- 카메라 연결 끊김 처리
- 광각 undistortion 옵션
- 정식 실행 로그

### 단계 6. 파지 후보 선택

- mask connected component별 후보 분리
- distance transform 기반 안전 내부점
- 손과 후보의 거리
- 손 접근 방향과 후보 방향 정렬
- 파지 가능한 실제 폭
- functional_region과의 안전거리
- 낮은 confidence 또는 애매한 후보에서 `ALIGN`

### 단계 7. 파지 자세 선택

| 후보 형태 | 파지 자세 |
|---|---|
| 가늘고 좁음 | PRECISION |
| 손잡이·중간 폭 원통 | WRAP |
| 넓은 컵·병 몸통 | POWER |

실제 폭 기준과 액추에이터 목표값은 Brunel Hand 실측 후 확정한다.

### 단계 8. ArUco 및 하드웨어 통합

1. ArUco로 손 위치와 접근 방향 계산
2. 작업대 마커로 픽셀-실거리 변환
3. Arduino serial 통신 mock test
4. 세 파지 프리셋 독립 구동
5. timeout과 `STOP` 검증
6. 비전 결과와 Brunel Hand 통합

---

## 19. 주요 산출물 위치

### 정책과 문서

- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `CLAUDE_CODE_HANDOFF.md`
- `docs/data_pipeline.md`
- `docs/contain_background_experiment.md`
- `docs/server_training.md`

### 공개 데이터

- `data/interim/affgrasp.jsonl`
- `data/interim/umd.jsonl`
- `data/processed/yolo_public`
- `data/processed/yolo_public_contain_background`

### 자체 데이터

- 원본: `data/raw/custom/images`
- 최종 수동 검수: `outputs/custom_mask_review/full_140_round2`
- 승인 mask: `data/interim/custom_approved_v1`
- 최종 YOLO: `data/processed/yolo_custom_approved_v1`

### 공개 학습 checkpoint

- `outputs/training/public_pretrain_d_contain_background_seed42/weights/best.pt`

### 자체 학습 설정

- `configs/training/custom_finetune_a_coco.yaml`
- `configs/training/custom_finetune_d_public.yaml`

### 평가 결과

- `outputs/evaluation/public_pretrain_d_seed42_test`
- `outputs/evaluation/public_pretrain_d_contain_background_seed42_original_test`
- `outputs/evaluation/public_pretrain_d_contain_background_seed42_expanded_test`

---

## 20. 개발 과정에서 확인된 사항

### 1. 일반 mAP만으로 프로젝트 정책 오류를 발견하기 어렵다

첫 모델은 mask mAP50 약 0.76이었지만 컵 contain 영역을 거의 전부
`functional_region`으로 잡았다. 연구 목적과 연결된 별도 안전·정책 지표가 필요하다.

### 2. ignore 정책은 모델 형식과 함께 설계해야 한다

통합 PNG에 ignore를 올바르게 저장해도 YOLO polygon 학습에서 ignore 샘플을 전부
제외하면 의도하지 않은 데이터 공백이 생긴다.

### 3. pseudo-label은 초기 후보일 뿐이다

Grounding DINO와 SAM2도 손잡이 누락, 몸통 혼동, 가림 각도에서 많은 오류를 냈다.
최종 validation/test mask는 사람이 검수해야 한다.

### 4. 같은 의미 클래스라도 파지 후보는 분리해야 한다

머그 손잡이와 몸통은 모두 grasp지만 파지 위치와 손 자세가 다르다. 의미 클래스만
합치는 것과 파지 후보를 분리하는 것은 별개의 문제다.

### 5. 실제 배경 negative가 중요하다

공개 데이터에서만 학습한 모델은 실제 카메라에서 책상과 벽을 크게 오검출했다. 실제
배경만 있는 이미지를 학습에 넣어야 false positive를 줄일 수 있다.

### 6. 프레임 수보다 실제 물체 수가 중요하다

같은 컵의 유사 프레임을 train과 test에 나누면 성능이 과대평가된다. 실제 물체 ID
기준 분할이 필수다.

### 7. 모델 출력 형식의 표현 한계를 감사 로그로 남겨야 한다

구멍이 있는 mask나 다중 component를 무리하게 polygon으로 바꾸면 정답 의미가 달라질
수 있다. 제외 이유, 정리 픽셀 수와 원본 대비 IoU를 기록해야 재현 가능하다.

---

## 21. 인계 후 즉시 수행할 작업

다음 작업자는 먼저 `AGENTS.md`와 `PROJECT_CONTEXT.md`를 읽고 라벨 정책과 물체 ID 기반 분할 기준을 확인해야 한다. 이후 로컬 단위 테스트 34개와 자체 YOLO 데이터의 train 110장, validation 50장, polygon 266개 구성을 다시 검증한다. 수치가 다르면 학습을 시작하지 않고 데이터 버전과 경로부터 확인한다.

서버에서는 기존 tmux 세션과 GPU 사용 상태를 확인한 뒤 지정 conda 환경을 활성화한다. 과거 smoke 또는 전체 학습 결과 폴더가 남아 있는지도 먼저 점검해 기존 결과를 덮어쓰지 않도록 한다.

실행 순서는 실험 D 1 epoch smoke test, 실험 A 1 epoch smoke test, smoke 결과 확인 순이다. 두 실험에서 데이터 로더, CUDA, 빈 배경 라벨 처리와 loss의 NaN/Inf 여부가 정상일 때 D 전체 학습을 먼저 시작한다. 전체 학습은 초반 2~3 epoch까지만 직접 확인하고 이후에는 tmux 세션에서 계속 실행한다.

---

## 22. 결론

공개 Affordance 데이터의 통합 변환, 사전학습, 고정 test 평가와 정책 검증을 완료했다. 외부 카메라 시험에서 확인된 배경 오검출을 보완하기 위해 실제 사용 환경에서 머그컵 3개와 빈 배경을 촬영했으며, 총 160장 규모의 자체 승인 데이터셋을 구축했다. 객체 이미지 140장의 손잡이와 몸통 마스크는 전량 수동 검수를 거쳐 YOLO 학습 형식으로 변환했다.

초기 공개 모델의 mask mAP50은 0.7600이었으나, 정책 감사 대상 117장에서 `contain` 영역을 모두 `functional_region`으로 예측하는 문제가 확인됐다. contain-background 변환 및 재학습 결과 해당 오검출은 117건에서 0건으로 감소했다. 일반 test 지표는 소폭 하락했지만 프로젝트의 컵 몸통 파지 정책을 충족하므로 이 checkpoint를 실험 D의 초기 가중치로 채택했다.

서버 환경, 데이터 bundle, A/D 학습 설정과 dry-run 검증은 완료됐다. 다음 단계는 두 실험의 1 epoch smoke test와 전체 미세조정이다. 자체 최종 test는 `mug_04` 촬영과 수동 정답 검수가 끝난 뒤 수행한다. 따라서 현 단계의 자체 데이터 결과는 validation으로만 해석하며 최종 unseen-instance 성능으로 보고하지 않는다.
