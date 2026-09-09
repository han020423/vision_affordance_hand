# 프로젝트 인수인계 문서

## 1. 최종 프로젝트 정의

### 국문 가제

**접근 방향과 물체 Affordance를 고려한 비전 기반 적응형 로봇손 파지 시스템**

### 영문 가제

**Vision-Based Adaptive Robotic Hand Grasping Considering Object Affordances and Hand Approach**

### 한 문장 목표

외부 RGB 카메라 영상에서 물체의 `grasp_region`과 `functional_region`을 분할하고, 로봇손의 접근 위치·방향에 가장 적합한 파지 후보를 선택한 뒤 Brunel Hand를 `Precision`, `Wrap`, `Power` 중 하나의 자세로 구동한다.

### 연구 성격

이 프로젝트의 핵심은 새로운 Segmentation 네트워크를 제안하는 것이 아니다. 다음 요소를 하나의 실제 시스템으로 연결하고 실험적으로 검증하는 응용·시스템 연구다.

1. Aff-Grasp와 UMD 기반 Affordance 학습
2. Grounding DINO와 SAM2를 이용한 자체 데이터 반자동 라벨링
3. 경량 Segmentation 모델을 이용한 실시간 추론
4. 손의 접근 위치·방향을 고려한 복수 파지 후보 선택
5. 후보 형상에 따른 3종 파지 선택과 실제 Brunel Hand 구동

---

## 2. 시스템 시나리오

```text
외부 고정 RGB 카메라
        ↓
YOLO Segmentation
        ↓
grasp_region / functional_region 마스크
        +
손목 ArUco 마커 검출
        ↓
손 위치·접근 방향과 후보별 특징 계산
        ↓
안전하고 접근하기 쉬운 파지 후보 선택
        ↓
Precision / Wrap / Power 결정
        ↓
Jetson → USB Serial → Arduino Nano 33 IoT
        ↓
DRV8833 → PQ12 액추에이터 → Brunel Hand
```

대표 시연은 머그컵으로 구성한다.

- 손이 손잡이 쪽에서 접근: 손잡이 후보 선택 → `Wrap`
- 손이 몸통 쪽에서 접근: 몸통 후보 선택 → `Power`
- 손이 가위 날 또는 망치 머리 쪽으로 접근: 기능 영역과의 충돌을 피하고 손잡이 후보로 유도

로봇팔을 이용한 완전 자동 접근은 범위에서 제외한다. 사용자가 로봇손을 물체 근처까지 움직이고 시스템이 파지 위치와 손 모양을 결정하는 반자율 방식이다.

---

## 3. 연구 질문과 가설

### 연구 질문

1. Aff-Grasp와 UMD로 사전학습한 뒤 자체 데이터로 미세조정하면, 자체 데이터만 학습한 모델보다 미학습 물체 인스턴스의 Affordance 분할 성능이 향상되는가?
2. 가장 가까운 영역만 선택하는 방식보다 접근 방향·크기 적합도·기능 영역 안전거리를 함께 고려할 때 올바른 파지 후보 선택률이 향상되는가?
3. 고정 `Power` 자세보다 영역 형상에 따라 `Precision`, `Wrap`, `Power`를 바꾸는 방식의 실제 파지 성공률이 높은가?

### 주장 범위

- 목표는 **학습에 사용하지 않은 실제 물체 인스턴스**와 제한된 미학습 범주에 대한 일반화다.
- 모든 임의 물체를 다루는 완전한 open-world 또는 zero-shot 시스템이라고 주장하지 않는다.
- `functional_region`은 물리적으로 절대 잡을 수 없는 영역을 뜻하지 않는다. 현재 작업에서 기능 보존과 안전을 위해 피해야 하는 영역으로 사용한다.

---

## 4. 하드웨어와 작업 환경

| 구성 | 역할 |
|---|---|
| 외부 USB RGB 카메라 | 물체와 로봇손 촬영 |
| Jetson | Segmentation 추론, 후보 선택, 상위 제어 |
| Brunel Hand v2.0 | 실제 파지 실행 |
| PQ12-100-12-P | 손가락 구동 |
| Arduino Nano 33 IoT | 액추에이터 저수준 제어 |
| DRV8833 | 액추에이터 구동 |
| 손목/손등 ArUco 마커 | 손의 2차원 위치와 접근 방향 추정 |
| 작업대 기준 ArUco 마커 | 픽셀-실거리 환산 및 좌표 기준 |
| 10 V 전원장치 | PQ12 전원 공급 |

카메라는 손에 부착하지 않고 작업 공간 외부에 고정한다.

- 테이블을 내려다보는 사선 시점 사용
- 물체와 손이 함께 보이는 약 30×30 cm 작업 영역 설정
- 카메라 위치, 초점, 해상도는 데이터 수집과 평가 중 고정
- 본 프로젝트의 접근 방향은 영상 평면에서 측정한 2차원 상대 위치와 방향이다.
- RGB-D 기반 6-DoF grasp pose 생성은 후속 연구 범위다.

---

## 5. 데이터셋 구성

최종 학습 데이터는 **Aff-Grasp + UMD + 자체 촬영 데이터**로 구성한다.

### 5.1 Aff-Grasp

역할:

- 정밀한 `graspable`/`functional` 의미 학습
- 실제 상호작용 및 복잡한 장면에 가까운 영상 분포 보완

사용 원칙:

- 공개 학습 데이터는 사전학습에 사용한다.
- Affordance Evaluation Dataset(AED)은 외부 평가용으로 유지하고 학습에 섞지 않는다.
- 공개 저장소의 파일 수와 실제 고유 샘플 수가 다를 수 있으므로, 다운로드 후 이미지-라벨 대응 관계와 중복을 스크립트로 검증한다.
- 공개 뷰어 구조상 학습 고유 샘플은 약 331개로 추정되지만, 코드에서 고정 숫자로 가정하지 않는다.

라벨 변환:

| 원본 라벨 | 통합 라벨 |
|---|---|
| graspable | `grasp_region` |
| functional | `functional_region` |

### 5.2 UMD RGB-D Part Affordance Dataset

역할:

- 다양한 생활도구 형상과 다중 시점 학습
- 동일 범주의 미학습 물체 인스턴스 평가 기반 제공

데이터 특성:

- 105개 실제 도구, 17개 물체 범주, 7개 affordance 라벨
- 전체 약 30,000 RGB-D 프레임
- 10,000장 이상에 수동 GT가 제공되는 것으로 안내됨
- 본 프로젝트에서는 RGB와 affordance mask만 사용한다.

사용 원칙:

- 원본 실험처럼 사람이 검수한 GT 프레임을 우선 사용한다.
- 자동 생성 라벨은 기본 학습에서 제외한다.
- 회전판 연속 촬영으로 생긴 유사 프레임이 많으므로 물체 ID와 시점 간격을 기준으로 3,000~5,000장 정도를 서브샘플링한다.
- Train/Validation/Test는 프레임이 아니라 **실제 물체 ID 기준**으로 나눈다.

라벨 변환:

| UMD 라벨 | 통합 라벨 | 비고 |
|---|---|---|
| grasp | `grasp_region` | 사용 |
| wrap-grasp | `grasp_region` | 사용 |
| cut | `functional_region` | 사용 |
| scoop | `functional_region` | 사용 |
| pound | `functional_region` | 사용 |
| support | `functional_region` | 사용 |
| contain | `ignore` 우선 | 컵 몸통 정책과 충돌 가능 |

`contain`을 무조건 `functional_region`으로 바꾸면 머그컵 몸통을 잡을 수 있는 후보로 사용하는 자체 정책과 충돌한다. 따라서 컵·머그의 몸통 `contain` 픽셀은 기본적으로 `ignore=255`로 두고, 자체 데이터에서 몸통 파지를 학습한다. 라벨이 중첩되거나 의미가 모호한 픽셀도 `ignore`로 처리한다.

### 5.3 자체 데이터

권장 최소 구성:

- 6개 범주
- 범주당 서로 다른 실제 물체 4개
- 물체당 20~25장
- 총 480~600장

우선 대상 범주:

1. 머그컵
2. 병 또는 원통형 용기
3. 망치 또는 고무망치
4. 가위
5. 프라이팬 또는 손잡이 용기
6. 주걱·국자 또는 유사 도구

촬영 변화:

- 물체 회전과 위치
- 카메라와의 거리
- 조명과 배경
- 손잡이가 보이는 정도
- 부분 가림
- 일부 손/ArUco 접근 장면

권장 물체 분할:

- 범주당 물체 2개: Train
- 범주당 물체 1개: Validation
- 범주당 물체 1개: Unseen-instance Test

같은 실제 물체의 다른 프레임을 Train과 Test에 나누지 않는다. Validation과 Test 라벨은 모두 사람이 최종 검수한다. 시간이 부족하면 각 분할에서 60~80장을 핵심 정답 세트로 먼저 완성한다.

### 5.4 최종 학습 규모

- Aff-Grasp 학습 샘플: 다운로드 후 검증, 약 331개 예상
- UMD 수동 GT 서브셋: 약 3,000~5,000장
- 자체 데이터: 약 480~600장
- 실제 학습·검증에 다루는 총량: 대략 3,800~5,900장

UMD가 데이터 대부분을 차지하므로 Aff-Grasp를 2~3배 오버샘플링하거나 데이터셋 균형 샘플러를 사용한다.

---

## 6. 최종 라벨 정책

모델의 의미 클래스는 다음 두 개다.

```yaml
0: grasp_region
1: functional_region
```

배경은 암묵적 background이고, 학습에서 판단하기 어려운 픽셀은 `ignore=255`로 유지한다.

핵심 규칙:

- 한 물체에 여러 파지 후보가 있으면 같은 `grasp_region` 클래스라도 별도 마스크/컴포넌트로 유지한다.
- 머그컵 손잡이와 몸통을 하나의 마스크로 합치지 않는다.
- 마스크는 단순 부품 전체보다 실제 손 접촉이 가능한 영역을 우선 표시한다.
- 기능 영역은 현재 작업에서 grasp 후보 점수를 낮추고 안전거리를 계산하는 데 사용한다.
- 배경 전체를 `functional_region`으로 라벨링하지 않는다.
- 공개 데이터의 의미가 자체 정책과 맞지 않으면 억지로 변환하지 말고 `ignore` 처리한다.

YOLO의 폴리곤 형식이 `ignore` 픽셀 또는 겹치는 조밀 라벨을 충분히 표현하지 못하면 다음 두 단계로 운영한다.

1. 원본 통합 마스크는 PNG semantic/instance mask 형태로 보존한다.
2. YOLO 학습용 변환본에서는 모호한 객체를 제외하고, 변환 로그에 제외 이유를 기록한다.

---

## 7. 자체 데이터 반자동 라벨링

Grounding DINO와 SAM2는 affordance 자체를 이해하는 정답 생성기가 아니라 라벨링 보조 도구로 사용한다.

```text
RGB 이미지
  ↓
구체적인 부품 텍스트 프롬프트
  ↓
Grounding DINO box
  ↓
SAM2 mask
  ↓
자동 품질 필터
  ↓
Accept / Human Review / Reject
  ↓
사람의 수정 및 최종 통합 라벨 저장
```

프롬프트 예시:

- `mug handle`, `mug body`
- `hammer handle`, `hammer head`
- `scissors handles`, `scissors blades`
- `pan handle`, `pan body`

자동 품질 검사 후보:

- Grounding DINO confidence
- SAM2 predicted IoU 또는 stability
- 마스크 면적 비율
- 연결 성분 수
- 영상 경계 접촉 여부
- `grasp_region`과 `functional_region`의 비정상적 중첩
- 인접 영상 간 마스크 일관성
- 서로 다른 프롬프트 결과의 충돌

초기 품질 점수 기준은 다음처럼 시작하되 Validation 데이터로 조정한다.

- `Q >= 0.80`: 자동 승인 후보
- `0.55 <= Q < 0.80`: 사람 검수
- `Q < 0.55`: 재생성 또는 폐기

보고서에서는 이를 **완전 자동 라벨링**이 아니라 **Foundation Model 기반 반자동 pseudo-labeling 및 자동 품질 필터링**으로 표현한다.

---

## 8. 모델 학습 계획

### 8.1 기본 모델

- 실시간 적용 모델: `YOLO11n-seg`
- 초기 입력 크기: `640×640`
- 최종 목표 장치: Jetson
- Grounding DINO와 SAM2는 오프라인 라벨 생성에만 사용

### 8.2 학습 단계

#### 단계 A: 데이터 변환 검증

1. 각 원본 데이터셋 파서 구현
2. 통합 semantic mask 생성
3. YOLO instance polygon 변환
4. 원본/변환 마스크 오버레이 시각화
5. 클래스별 최소 30장 수동 점검
6. 물체 ID 기반 split 누수 검사

#### 단계 B: 공개 데이터 사전학습

- Aff-Grasp + UMD 통합 데이터 사용
- 60~80 epochs에서 시작
- learning rate 약 `0.001`에서 시작
- Aff-Grasp 오버샘플링 또는 balanced sampler 적용
- 성능이 정체되면 early stopping

#### 단계 C: 자체 데이터 미세조정

- 공개 데이터 사전학습 가중치에서 시작
- 자체 Train 데이터로 80~120 epochs
- learning rate 약 `0.0003`에서 시작
- Validation 성능 기준 early stopping patience 20~25
- 색상, 밝기, 회전, 크기, 평행이동 증강 사용
- 마스크 의미를 훼손하는 과도한 crop이나 mix augmentation은 시각적으로 검증 후 사용

위 수치는 확정값이 아니라 첫 실행값이다. 실제 GPU 메모리와 Validation 곡선에 따라 batch size와 epoch를 조정한다.

### 8.3 필수 비교 실험

| 실험 | 초기 가중치/사전학습 | 미세조정 | 목적 |
|---|---|---|---|
| A | COCO pretrained | 자체 데이터 | 자체 데이터만 사용한 기준선 |
| B | Aff-Grasp | 자체 데이터 | Aff-Grasp 기여도 |
| C | UMD | 자체 데이터 | UMD 기여도 |
| D | Aff-Grasp + UMD | 자체 데이터 | 최종 제안 학습 방식 |

시간이 부족하면 최소한 `A`와 `D`는 동일한 조건으로 비교한다.

---

## 9. 추론 결과와 파지 후보 표현

한 물체에서 분리된 `grasp_region` 마스크 또는 연결 성분을 각각 독립 후보로 유지한다.

```python
candidate = {
    "mask": None,
    "confidence": 0.0,
    "safe_point_px": (0, 0),
    "nearest_point_px": (0, 0),
    "width_mm": 0.0,
    "orientation_rad": 0.0,
    "aspect_ratio": 0.0,
    "functional_clearance_mm": 0.0,
}
```

파지점은 단순 centroid만 사용하지 않는다. 마스크 내부 distance transform의 최댓값을 이용해 경계에서 가장 멀리 떨어진 안전한 점을 구하고, 손과의 거리 및 실제 접근 가능성을 함께 고려한다.

손의 상태는 ArUco로 다음처럼 표현한다.

```text
p_hand = (x, y)       # 영상 평면 손 위치
v_hand = (vx, vy)     # 손이 향하는 방향
```

---

## 10. 접근 방향 기반 후보 선택

각 파지 후보의 점수는 다음 요소로 계산한다.

```text
S_i = 0.30 C_i
    + 0.25 D_i
    + 0.20 A_i
    + 0.15 F_i
    + 0.10 M_i
```

- `C_i`: Segmentation confidence
- `D_i`: 손과 후보 사이 거리 점수
- `A_i`: 손 방향과 후보 방향의 정렬 점수
- `F_i`: 후보 폭·형상과 Brunel Hand 개구 범위의 적합도
- `M_i`: `functional_region`으로부터의 안전 여유 거리

가중치는 설명 가능한 초기값이며 Validation 실험으로 조정한다. 결과 보고 시 최종 가중치와 조정 근거를 기록한다.

다음 후보는 거부하거나 사용자 재정렬을 요구한다.

- 신뢰도가 낮은 후보
- 손의 최대 개구 폭보다 큰 후보
- 손의 진행 방향에서 크게 벗어난 후보
- 영상 경계에 걸려 잘린 후보
- 기능 영역과 안전 마진 없이 겹치는 후보
- 손에 심하게 가려져 일정 시간 안정적으로 추적되지 않은 후보

후보 간 점수 차이가 작으면 즉시 파지하지 않고 화면에 정렬 방향을 표시한다.

---

## 11. 파지 유형 결정

Feix의 인간 파지 taxonomy 전체를 직접 분류하지 않는다. Brunel Hand에서 구현 가능한 세 가지 시너지로 축소한다.

| 파지 | 적용 기준 | 예시 |
|---|---|---|
| `PRECISION` | 좁고 가는 후보, 높은 종횡비 | 펜, 숟가락 손잡이 |
| `WRAP` | 중간 폭의 손잡이·원통 | 머그컵 손잡이, 망치 손잡이 |
| `POWER` | 넓은 몸통 | 컵 몸통, 병 |

초기 규칙:

```python
if width_mm <= T_precision and aspect_ratio >= R_precision:
    grasp_type = "PRECISION"
elif width_mm <= T_wrap:
    grasp_type = "WRAP"
else:
    grasp_type = "POWER"
```

`T_precision`, `T_wrap`, 최대 개구 폭과 각 액추에이터 목표값은 Brunel Hand 실측과 반복 파지로 보정한다. 물체 클래스 이름을 직접 이용해 자세를 고정하는 규칙은 기준선 외에는 사용하지 않는다.

---

## 12. Brunel Hand 제어

Jetson에서 Arduino로 전달할 최소 명령:

```text
OPEN
PRECISION
WRAP
POWER
STOP
```

상태 머신:

```text
SEARCH
  → TARGET_SELECTED
  → ALIGN
  → PRE_SHAPE
  → CLOSE
  → HOLD
  → OPEN
```

필수 안전 조건:

- 최대 동작시간 제한
- 위치 피드백 허용 범위 확인
- 통신 중단 시 정지
- 사용자 `STOP` 우선 처리
- 파지 직전 대상 마스크 소실 시 정지
- PQ12 피드백이 불안정하면 50~100 ms 펄스 구동과 하드 타임아웃 사용

모델과 손 제어를 처음부터 동시에 개발하지 않는다. 먼저 세 파지 프리셋을 독립적으로 안정화한 후 비전 결과와 연결한다.

---

## 13. 평가 계획

### 13.1 Segmentation

- Mask mAP50
- Mask mAP50-95
- 클래스별 IoU 또는 Dice/F1
- Seen category / unseen physical instance 비교
- 가능하면 제한된 unseen category 비교
- Jetson latency와 FPS

추가 안전 지표:

```text
UnsafeOverlap =
|predicted grasp_region ∩ GT functional_region|
------------------------------------------------
|predicted grasp_region|
```

낮을수록 현재 작업에서 안전한 예측이다.

### 13.2 후보 선택

- 정답 파지 후보 선택률
- 손의 시작 위치·접근 방향별 선택률
- 후보 전환의 안정성
- 정렬 완료까지 걸린 시간
- 기능 영역 쪽에서 접근했을 때 안전 후보로 유도한 비율

### 13.3 실제 파지

성공 기준은 시험 전에 고정한다.

> 물체를 테이블에서 5 cm 이상 들어 올리고 3초 이상 떨어뜨리지 않고 유지하면 성공

비교 방법:

| 방법 | 후보 선택 | 파지 자세 |
|---|---|---|
| Baseline 1 | 사용자가 지정 또는 고정 위치 | 고정 `POWER` |
| Baseline 2 | 가장 가까운 후보 | 고정 `POWER` |
| Proposed | 거리+방향+적합도+안전거리 | `PRECISION/WRAP/POWER` 적응 선택 |

각 물체와 접근 방향에서 동일 횟수로 반복하고 성공률과 실패 원인을 기록한다.

---

## 14. 3주 실행 일정

### 1주차: 데이터와 독립 모듈

- Aff-Grasp와 UMD 구조 확인 및 다운로드 기록
- 통합 라벨 변환기와 split 생성기 구현
- 변환 마스크 시각 검수 도구 구현
- 카메라와 ArUco 좌표계 설치
- 자체 물체 촬영 시작
- Grounding DINO + SAM2 반자동 라벨링 파이프라인 구축
- Brunel Hand의 최대 개구 폭 측정
- `Precision`, `Wrap`, `Power` 프리셋 독립 구동

완료 기준:

> 데이터 변환 예시가 정확히 시각화되고, 손의 세 자세가 명령별로 반복 구동된다.

### 2주차: 학습과 후보 선택

- 공개 데이터 사전학습
- 자체 데이터 미세조정
- A 대 D 최소 비교 실험
- 실시간 mask와 confidence 출력
- distance transform 기반 안전 파지점 계산
- ArUco 손 위치·방향 추정
- 후보 점수와 파지 자세 규칙 구현

완료 기준:

> 손 위치를 바꾸면 머그컵 손잡이/몸통 후보와 파지 자세가 실시간으로 전환된다.

### 3주차: 통합과 평가

- Jetson–Arduino Serial 통합
- 상태 머신과 비상정지 구현
- 미학습 인스턴스 Segmentation 평가
- Baseline과 제안 방법의 실제 파지 반복 실험
- 실패 사례 분류
- 표, 그래프, 시연 영상과 발표자료 제작

완료 기준:

> 미학습 물체에서 후보 선택과 적응형 파지를 시연하고 정량 결과를 제시한다.

---

## 15. 우선순위와 축소 기준

### 반드시 완성

1. 통합 라벨 변환과 검증
2. YOLO11n-seg 학습 및 자체 테스트
3. ArUco 기반 2D 접근 방향
4. 세 파지 프리셋
5. 머그컵 손잡이/몸통 적응형 시연
6. A 대 D 비교와 실제 파지 성공률

### 시간이 남으면

- 자동 품질 점수 정교화
- 제한된 unseen-category 평가
- Temporal mask smoothing
- Jetson 최적화 또는 TensorRT 변환

### 이번 프로젝트에서 제외

- Hailo 변환
- 로봇팔 자동 이동
- RGB-D 기반 6-DoF grasp pose
- Feix 33종 전체 구현
- VLM 기반 완전 open-world reasoning
- RAGNet 또는 Aff-Grasp 전체 네트워크의 완전 재현

---

## 16. 예상 산출물

- Aff-Grasp/UMD 통합 변환 스크립트
- 자체 Affordance 데이터셋과 라벨 검수 결과
- 학습 설정 및 모델 가중치
- 학습·평가 결과 표
- 실시간 Segmentation 프로그램
- ArUco 손 위치·방향 추적 모듈
- 접근성 기반 후보 선택 모듈
- Arduino/PQ12 세 파지 제어 코드
- 통합 시연 프로그램과 영상
- 실패 사례 분석 및 발표자료

---

## 17. 참고 연구와 공식 자료

- Aff-Grasp paper: <https://arxiv.org/abs/2408.10123>
- Aff-Grasp code: <https://github.com/Reagan1311/Aff-Grasp>
- Aff-Grasp public data: <https://huggingface.co/datasets/Gen1113/Data_for_Aff-Grasp>
- UMD Part Affordance Dataset: <https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index.html>
- Feix et al., *The GRASP Taxonomy of Human Grasp Types*: <https://doi.org/10.1109/TOH.2015.2470657>
- Grounding DINO: <https://github.com/IDEA-Research/GroundingDINO>
- SAM 2: <https://github.com/facebookresearch/sam2>

---

## 18. 아직 실측·확정해야 하는 항목

Codex가 임의로 정하지 말고 사용자 실험값 또는 첫 데이터 점검 후 확정해야 한다.

- 실제 Jetson 모델과 사용 가능한 VRAM
- 카메라 모델, 해상도, 프레임률, 설치 높이와 각도
- Brunel Hand의 최대 개구 폭
- `T_precision`, `T_wrap`, `R_precision`
- 파지 프리셋별 PQ12 목표 위치 또는 펄스 시간
- ArUco 마커 크기와 ID
- UMD 다운로드 후 실제 사용 가능한 수동 GT 수
- Aff-Grasp 파일 구조와 실제 고유 샘플 수
- 후보 점수의 최종 가중치와 rejection threshold

---

## 19. Codex에 처음 전달할 명령

```text
저장소 루트의 AGENTS.md와 PROJECT_CONTEXT.md를 먼저 전부 읽어.
프로젝트의 라벨 정책과 실험 분할을 임의로 바꾸지 마.

우선 저장소를 점검한 뒤 다음 작업만 수행해.
1. 권장 디렉터리 구조를 제안한다.
2. Aff-Grasp/UMD 원본을 통합 라벨로 변환하는 설계를 작성한다.
3. 데이터셋 누수와 라벨 변환을 검증하는 테스트 계획을 작성한다.
4. 구현 전에 필요한 실제 경로와 환경 정보만 질문한다.

아직 데이터가 없으면 가짜 데이터나 결과를 만들지 말고,
다운로드 위치와 예상 파일 구조를 설정값으로 분리한 코드 뼈대만 작성해.
```
