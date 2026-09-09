# 파지 자세 결정의 기하적 근거: EPIC-KITCHENS VISOR 868건 분석

작성일: 2026-08-24
데이터: `outputs/visor_geometry/features.csv` (868건), 그림: `outputs/visor_geometry/grasp_geometry_evidence.png`
재현: `python scripts/visor_grasp_geometry.py`

## 1. 문제

기존 자세 결정은 후보 마스크의 내접원 지름을 픽셀 임계값(40 px)과 비교했다. 이 방식은

- 실측 근거가 없고(임의 초기값),
- 픽셀 폭이 카메라 거리에 좌우되어 같은 물체도 거리에 따라 자세가 뒤집혔다.
  실제로 가위·드라이버 자루가 WRAP으로, 작은 머그 손잡이가 PRECISION으로 오분류됐다.

지도교수 요구는 "파지 자세의 근거"다. 이 분석은 사람의 실제 파지 행동에서 그 근거를 추출한다.

## 2. 방법

기존에 수행한 EPIC-KITCHENS VISOR 파지 판정(868장면, 판정자 1인, `docs/human_grasp_statistics.md`)을
원본 VISOR 주석(`data/raw/visor/annotations`, 829 MB)에 다시 연결해 각 장면의 **손 마스크와
물체 마스크**를 복원했다(868/868건 전부 연결 성공). Feix의 방법론(물체 치수와 파지 유형의
상관 분석, 약 10,000회 관찰)을 따르되, VISOR의 분할 마스크 덕분에 두 가지를 추가로 할 수 있었다.

1. **손 크기 정규화**: 모든 치수를 손바닥 폭(손 마스크 distance transform 최댓값×2)으로 나눔.
   비율은 카메라 거리와 무관하므로 실행 시 같은 특징을 쓸 수 있다.
2. **접촉부 국소 치수**: 손 주변 접촉 대역에서 물체의 국소 두께를 측정
   (Feix의 "grasped dimension"에 해당).

표본별 특징: 접촉부 두께비(r_contact), 물체 단축비(r_minor), 물체 종횡비(obj_aspect),
부속성(appendage = 접촉부 두께 / 물체 최대 두께). 판정 분포: handle 439 · body 178 ·
not_grasp 129 · rim_pinch 114 · functional_touch 8.

## 3. 결과

### 발견 1 — 잡은 부위의 폭은 파지 유형을 가르지 못한다

| 판정 | n | r_contact 중앙값 (25~75%) |
|---|---|---|
| handle 파지 | 439 | 0.154 (0.114~0.214) |
| body 파지 | 178 | 0.171 (0.130~0.210) |
| rim_pinch | 114 | 0.192 (0.155~0.231) |

세 분포가 거의 완전히 겹친다(1차원 경계의 균형정확도 0.50 = 무작위 수준).
**사람은 어떤 물체든 손바닥 폭의 15~19% 두께인 부위를 골라 잡는다.** 이는 Feix의
"파지 지점의 96%가 7 cm 이하" 관찰과 일치하며, 잡는 부위의 폭이 거의 일정하기 때문에
폭 임계값으로 자세를 결정하는 것은 원리적으로 근거가 없다는 뜻이다.
기존 40 px 규칙이 실패한 근본 이유다.

### 발견 2 — 자세를 가르는 것은 물체의 구조다

handle 판정 418건을 물체군으로 나누면(도구 자루: knife·scissors·spoon·ladle·spatula·fork 252건,
용기의 부속 손잡이: mug·cup·kettle·pan·pot 166건):

| 특징 | 도구 자루 중앙값 | 용기 손잡이 중앙값 | 최적 경계 | 균형정확도 |
|---|---|---|---|---|
| 물체 전체 종횡비 | 4.18 | 1.30 | **2.12** | **0.936** |
| 부속성(접촉부/최대 두께) | 0.52 | 0.20 | 0.27 | 0.805 |
| 접촉부 두께비 | 0.141 | 0.180 | — | 0.50 (무의미) |

도구는 물체 전체가 길쭉하고(자루가 곧 물체), 용기 손잡이는 둥근 큰 몸통에 붙은
가는 부속이다. 이 구조 차이가 로봇의 자세 구분과 정확히 대응한다:
도구 자루·펜 → 집기(PRECISION), 용기의 부속 손잡이 → 걸어 감기(WRAP).

### 발견 3 — rim_pinch는 "작은 물체"가 아니라 "크고 둥근 물체"에서 나온다

rim_pinch(접시·뚜껑·도마의 테두리 집기)의 물체 단축비 중앙값은 2.05로 handle(0.77)·body(0.87)보다
훨씬 크다. 집기가 작은 물체에서 나온다는 직관은 2D 투영에서 성립하지 않는다
(접시의 "얇음"은 화면에 보이지 않는 3차원 치수다). 크기 기반 PRECISION 판정이
위험한 또 하나의 이유다.

## 4. 도출한 자세 결정 규칙

`src/grasp_selection/scoring.py`의 `_decide_pose()`로 구현했다. 모든 특징이
비율·인접성이라 **카메라 거리·mm 보정과 무관**하다.

```text
1) body 후보                         → POWER
2) 손잡이 후보가 body 영역과 인접      → 용기의 부속 손잡이 → WRAP     (근거: 발견 2, 부속성 0.805)
3) 그 외: 물체 문맥(손잡이 ∪ 인접 기능영역)의 종횡비
     ≥ 2.1  (VISOR 경계 2.12)        → 도구 자루/펜 → PRECISION       (근거: 발견 2, 0.936)
     < 2.1                           → WRAP (기본)
```

오분류 사례에 대한 효과:

- **가위·드라이버**: 몸통 영역이 없고, 손잡이∪날(기능영역)이 길쭉 → PRECISION ✓
- **작은 머그 손잡이**: 몸통 영역과 인접 → WRAP ✓ (크기·거리와 무관)

세 프리셋을 셋으로 둔 1차 이유는 손 하드웨어다. Brunel Hand의 액추에이터는 검지, 중지,
약지·소지 묶음, 엄지 네 개뿐이고 약지와 소지는 따로 움직이지 않는다. 검지와 중지는 어느
자세에서나 참여하므로 실제로 갈리는 축은 약지·소지 참여 여부와 엄지 사용 여부 둘이고,
조합은 넷이다. 그중 둘 다 빼는 조합은 파지력이 나오지 않아 쓰지 않으므로 셋이 남는다.
힘 제어가 없고 스톨로 정지하므로 세기를 조절해 자세를 더 나눌 수도 없다.

분류학적으로 보면 이 셋은 Feix의 GRASP Taxonomy(33종)를 배열하는 축 위에 놓인다. Feix는
33종을 대립 유형(Iberall)과 엄지 위치로 배열하는데, PRECISION은 pad opposition,
POWER는 palm opposition에 대응하고, WRAP은 엄지를 쓰지 않는 갈고리형이라 대립 유형
바깥의 별도 항목이다. Iberall의 대립은 pad·palm·side 셋이며 우리 셋과 일대일로 대응하지
않는다는 점은 명시해 둔다. 33종을 더 잘게 구분해도 이 손에서는 같은 구동 명령이 된다.

## 4.1 통계 검증 (2026-08-24 추가)

종횡비 판별기(도구 자루 vs 용기 손잡이, n=418)에 대해:

- 5-겹 교차검증 ×20회: 균형정확도 0.927 ± 0.025 (표본 내 0.936과 근접 — 과적합 없음)
- 부트스트랩 2000회: 95% 신뢰구간 [0.929, 0.936]
- 학습된 경계의 안정성: 1.94~2.26 (중앙값 2.15)
- 물체별 hold-out(해당 물체를 제외하고 학습 후 분류 — 미학습 물체 일반화의 모사):
  mug·cup·kettle·fork 1.00 / ladle 0.96 / knife 0.94 / pan 0.91 / spoon 0.90 /
  spatula 0.88 / pot 0.85 / scissors 0.82 (n≤3인 bottle·jar 제외)

## 5. 근거의 성격 구분 (중요)

이 분석이 뒷받침하는 것과 뒷받침하지 않는 것을 구분한다.

- **사람 데이터가 직접 뒷받침**: (a) 잡은 부위 폭은 자세 결정 기준이 될 수 없다,
  (b) 물체 종횡비·부속성이 용기 손잡이와 도구 자루를 가른다.
- **로봇 설계 결정**: 도구 자루 → PRECISION 배정. 사람은 도구 자루를 주로 감아쥔다.
  이 배정의 근거는 Brunel Hand의 능력 제약(3지 집기가 가는 막대에 적합)이며,
  VISOR는 "두 물체군이 기하적으로 분리 가능함"까지를 증명한다.

## 5.1 한계

- 판정자 1인, 도구 자루의 인간 파지는 대부분 감아쥐기(wrap)다. 로봇에서 도구 자루를
  PRECISION으로 잡는 것은 Brunel Hand의 능력 제약(3지 집기가 가는 막대에 적합)에 따른
  설계 결정이고, VISOR 분석은 "어떤 기하가 물체군을 가르는가"의 근거로 쓴다.
- 2D 마스크 기반이므로 두께(깊이 방향)는 보지 못한다. rim_pinch류(접시)는 현재 시연
  범위 밖이며, 포함하려면 발견 3의 r_minor 특징을 추가해야 한다.
- 인접성 판정은 분할 모델이 몸통을 놓치면 3)의 기본 경로로 떨어진다. 머그는 몸통이
  크게 잡히므로 실용상 문제가 적다.

## 4.2 자체 도메인 검증 (2026-08-24, 사용자 실물 시험)

젯슨 + 고정 카메라 + RF-DETR 분할로 새 규칙을 실물 시험했다.

- 머그·도구류 등 시험 물체에서 **자세 판정이 거의 전부 의도대로** 동작했다
  (VISOR에서 도출한 경계가 자체 카메라 도메인에서도 유효함을 확인).
- **알려진 실패 1건: 투명 2 L 물병.** 분할 모델이 몸통을 `handle_grasp_region`으로
  오분류했고, 그 입력에 대해 자세 규칙은 "몸통 없음 + 길쭉한 단독 후보 → PRECISION"을
  냈다(기대: body → POWER. 사람 행동 통계에서도 병은 몸통 파지 92~100 %).

물병 건의 원인은 자세 규칙이 아니라 **상류의 분할 오류**다: 학습 데이터가 불투명 머그
3개 + 배경뿐이라 투명한 대형 용기는 분포 밖이고, 투명 물체 분할은 그 자체로 별도
연구 주제일 만큼 어려운 사례다. 잘못된 클래스가 들어오면 규칙은 그 클래스 기준으로
일관되게 동작하므로, 교정 지점은 규칙이 아니라 분할이다.

완화 방안: (a) 투명 용기 촬영본을 미세조정 데이터에 추가(배경 음성 20장으로 오검출을
줄인 것과 같은 방법), (b) 시연 물체 집합에서 투명 대형 용기를 제외.

## 6. 방법론의 선행 연구 대응

본 방법의 구성 요소는 각각 확립된 선행 연구가 있고, 이들의 조합과 실측(868건,
교차검증)이 본 작업의 기여다.

| 본 방법의 요소 | 선행 연구 | 분야 |
|---|---|---|
| 물체를 부분·구조로 나눠 자세 배정 (규칙 ①②) | Miller et al. 2003 (shape primitives), Vahrenkamp et al. 2016 (part-based grasp planning) | 로봇 파지 계획 |
| 기하 규칙으로 소수 프리셋 중 자동 선택, 사람이 접근을 맡는 반자율 분담 | Došen & Popović 2010 (cognitive vision system), Shi et al. 2025 (기하·대칭축 의수 파지), Hannes pre-shape 선택 2022 | 의수 제어 |
| 손 크기 정규화(무차원 비율)가 파지 형태를 결정 | Cesari & Newell 2000 (body-scaled transitions: 파지 형태 전환이 물체/손 크기 무차원 비율의 일정 임계값에서 발생, 유아~성인 공통) | 생태심리학 |
| 사람 관찰 통계에서 자세 기준 도출 | Feix et al. 2014 (약 10,000회 관찰), Bullock et al. 2015 (Yale grasping dataset), Yu et al. 2022 (egocentric 영상 affordance 정밀 주석) | 파지 행동 분석 |
| 세 자세의 분류학적 틀 | Iberall 1997 (opposition space: pad/palm/side 셋 중 pad·palm이 PRECISION·POWER에 대응), Feix 2016 (33종을 대립 유형과 엄지 위치로 배열) | 파지 분류학 |

## 7. 참고 문헌

- T. Feix, I. M. Bullock, A. M. Dollar, "Analysis of Human Grasping Behavior: Object Characteristics and Grasp Type," IEEE Trans. Haptics, 2014.
- T. Feix et al., "The GRASP Taxonomy of Human Grasp Types," IEEE THMS, 2016.
- I. M. Bullock, T. Feix, A. M. Dollar, "The Yale Human Grasping Dataset," IJRR 34(3), 2015.
- T. Iberall, "Human Prehension and Dexterous Robot Hands," IJRR, 1997 (opposition space).
- P. Cesari, K. M. Newell, "Body-scaled transitions in human grip configurations," JEP:HPP 26(5), 2000.
- A. T. Miller et al., "Automatic Grasp Planning Using Shape Primitives," ICRA, 2003.
- N. Vahrenkamp et al., "Part-based Grasp Planning for Familiar Objects," Humanoids, 2016.
- S. Došen, D. B. Popović et al., "Cognitive vision system for control of dexterous prosthetic hands," J. NeuroEng. Rehabil. 7:42, 2010.
- M. Stival(Hannes) et al., "Grasp Pre-shape Selection by Synthetic Training: Eye-in-hand Shared Control on the Hannes Prosthesis," IROS, 2022 (arXiv:2203.09812).
- Shi et al., "Vision-Based Grasping Method for Prosthetic Hands via Geometry and Symmetry Axis Recognition," Biomimetics 10(4):242, 2025.
- Z. Yu et al., "Precise Affordance Annotation for Egocentric Action Video Datasets," arXiv:2206.05424, 2022.
- A. Darkhalil et al., "EPIC-KITCHENS VISOR Benchmark," NeurIPS 2022 (데이터).
