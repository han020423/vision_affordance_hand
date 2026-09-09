# 작업 계획: RF-DETR 출력을 파지 후보 선택에 연결 (실험 M 어댑터)

작성: 2026-08-24. 이 문서는 독립 세션이 이 작업만 수행할 수 있게 쓴 계획서다.
시작 전에 `CLAUDE.md`, `AGENTS.md`를 읽고 라벨·문서 정책을 따른다.

## 1. 목표

배포 후보로 확정된 RF-DETR-Seg 모델(실험 M)의 실시간 출력을 기존 파지 후보
선택 파이프라인(`src/grasp_selection`: distance transform 안전 내부점, 폭 추정,
PRECISION/WRAP/POWER 결정)에 연결해, `run_realtime_rfdetr.py`에서 `--select`
모드(마우스=모의 손)가 동작하게 한다.

완료 기준:
1. `python scripts/run_realtime_rfdetr.py --select` 실행 시 마우스 커서를 모의
   손으로 삼아 후보 채점·선택 결과(GRASP/ALIGN/NO_TARGET, 자세)가 화면에 그려진다.
2. `--image <경로> --select` 단일 이미지 모드가 결정 JSON을 출력한다
   (`run_realtime_seg.py`의 같은 모드와 동일한 출력 형식).
3. 새 변환 함수의 단위 테스트가 추가되고 전체 테스트가 통과한다.
4. 기존 Ultralytics 경로(`run_realtime_seg.py`)는 수정하지 않는다.

## 2. 현재 상태 (검증된 사실)

- **모델**: `outputs/training/custom_finetune_m_rfdetr_customv3_seed42/checkpoint_best_ema.pth`
  (RFDETRSegNano, 3클래스). 운용 임계값 0.25.
- **rfdetr 예측 형식**: `model.predict(PIL_RGB, threshold=...)` →
  supervision `Detections`: `.mask` (N,H,W bool, 프레임 크기), `.class_id`,
  `.confidence`. **class_id는 0부터 시작**(0=handle, 1=body, 2=functional)이
  실측으로 확인됨. 다만 방어적으로 base 확인 로직(`run_realtime_rfdetr.py`의
  `class_id_base` 관측)을 유지할 것.
- **후보 추출 진입점이 이미 존재**:
  `src/grasp_selection/candidates.py::extract_candidates_from_arrays(masks,
  class_ids, confidences, frame_shape, *, min_confidence=0.35, min_area_px=400,
  boundary_margin_px=4)` — RF-DETR 등 비-Ultralytics 경로용으로 만들어져 있고,
  클래스 규약은 0 handle / 1 body / 2 functional. functional(2)은 후보가 아니라
  회피 마스크로 합쳐진다.
- **채점·결정**: `src/grasp_selection/scoring.py::decide_grasp(candidates,
  functional_mask, HandState, config)` → `Decision(state, candidate, pose, ...)`.
  설정은 `load_selection_config(configs/grasp_selection.yaml,
  configs/human_grasp_prior.yaml)`.
- **그리기 재사용**: `src/perception/realtime.py::draw_selection(overlay,
  decision, hand)` 는 모델 무관(Decision만 받음) — 그대로 재사용.
- **참조 구현**: `scripts/run_realtime_seg.py`의 `--select` 흐름(마우스 콜백,
  단일 이미지 모드 블록)이 완전한 예시다. rfdetr 버전은 이 흐름을 미러링한다.
- **알려진 이슈**: 실시간에서 맨손이 handle 0.25로 잠깐 오검출된 사례 있음
  (2026-08-24 실측). 후보 추출 기본 `min_confidence=0.35`가 이런 저신뢰 후보를
  이미 걸러준다 — 설정값을 낮추지 말 것.

## 3. 구현 단계

### 1단계 — 변환 함수 (작음)

`scripts/run_realtime_rfdetr.py`에 rfdetr Detections → 배열 3개(masks,
class_ids, confidences) 변환 함수를 추가한다. class_id_base를 빼서 0기준으로
정규화하고, `detections.mask is None`/빈 검출을 빈 배열로 처리한다.

### 2단계 — `--select` 실시간 모드

`run_realtime_rfdetr.py`의 루프에서 `--select`일 때:
1. 변환 함수 → `extract_candidates_from_arrays(...)`
   (min_confidence 등은 `load_selection_config` 결과의 `candidate` 절 사용 —
   `run_realtime_seg.py`와 동일한 키)
2. 마우스 콜백으로 `HandState(position=커서, direction=이동방향)` 구성
   (`src/perception/realtime.py::run_camera_loop`의 on_mouse 로직 참조)
3. `decide_grasp(...)` → `draw_selection(overlay, decision, hand)`
CLI 인자 `--select`, `--selection-config`, `--human-prior-config`는
`run_realtime_seg.py`와 같은 이름·기본값으로 맞춘다.

### 3단계 — `--image` 단일 이미지 모드

`run_realtime_seg.py`의 단일 이미지 `--select` 블록(모의 손 위치 `--hand-x/-y`,
화면 중심 향한 접근 방향 가정, 결정 JSON 출력, overlay 저장)을 rfdetr 버전으로
미러링한다. 출력 JSON 키(state/pose/reason/candidates)를 동일하게 유지해
두 모델의 결정을 비교할 수 있게 한다.

### 4단계 — 단위 테스트

`tests/test_rfdetr_selection_adapter.py`:
- 합성 마스크(손잡이/몸통/기능 사각형)로 변환 함수 → `extract_candidates_from_arrays`
  경로: 후보 2개(handle, body), functional은 회피 마스크로 감. 
- class_id_base=1 입력이 0기준으로 정규화되는지.
- 빈 검출·mask None 입력이 빈 후보를 반환하는지.
- 저신뢰(0.25) 후보가 min_confidence 0.35에서 걸러지는지(손 오검출 회귀).
변환 함수는 rfdetr import 없이 테스트 가능해야 한다(덕 타이핑: mask/class_id/
confidence 속성을 가진 간단한 객체를 넘김).

### 5단계 — 검증

1. `python -m unittest discover -s tests -p "test_*.py"` 전체 통과.
2. 단일 이미지 검증(카메라 불필요):
   `data/processed/rfdetr_custom_v3_test_mug04/valid/`의 mug_04 이미지 2~3장으로
   `--image ... --select --conf 0.25` 실행, GRASP 결정과 WRAP/POWER 자세가
   나오는지, overlay에서 선택 후보가 손잡이/몸통에 찍히는지 확인.
3. (사용자 참여 가능 시) 실시간 `--select`로 마우스 모의 손 동작 확인.

## 4. 하지 말 것 / 함정

- `src/grasp_selection`의 채점 로직·가중치·설정 스키마를 바꾸지 않는다
  (이 작업은 어댑터일 뿐이다). 가중치 합=1 검증이 있으니 설정 수정 금지.
- 기존 Ultralytics 경로와 검수·승인 데이터는 손대지 않는다.
- `min_confidence`(기본 0.35)를 낮추지 않는다 — 손 오검출(0.25) 필터 역할.
- mm 캘리브레이션 전에는 폭 적합도가 중립(0.5)으로 처리되는 것이 정상이다
  (`_width_fitness` 참조). 자세가 전부 같은 값으로 나와도 버그가 아니다.
- 로컬은 CPU 추론이라 3~5 FPS가 정상. 속도 최적화는 이 작업 범위 밖.
- 코드 주석·docstring·출력 문구는 한국어(호환 필요한 식별자는 영문).

## 5. 산출물

- 수정: `scripts/run_realtime_rfdetr.py` (변환 함수 + --select + --image)
- 추가: `tests/test_rfdetr_selection_adapter.py`
- 검증 로그: 단일 이미지 결정 JSON 2~3건 (`outputs/realtime_test/`)
- 이 문서 하단에 "구현 결과" 절을 추가해 완료 기록을 남긴다

---

## 구현 결과 (2026-08-24)

계획 1~5단계를 모두 수행했다. 로컬 Windows(CPU)에서 구현·검증했다.

### 변경 파일

- 수정: `scripts/run_realtime_rfdetr.py` (274 → 462줄)
  - `detections_to_arrays(detections, class_id_base=0)`: rfdetr Detections →
    (masks, class_ids, confidences) 변환. class_id_base를 빼서 0기준으로
    정규화하고, None/빈 검출은 길이 0 배열, 길이 불일치는 ValueError.
    mask/class_id/confidence 속성만 쓰므로(덕 타이핑) rfdetr 없이 시험 가능.
  - `--select` 실시간 모드: 마우스 콜백(EMA 접근 방향)은 `run_camera_loop`와
    동일한 로직을 미러링. `--no-window`와 함께 쓰면 모의 손을 화면 중앙에
    고정한다. 채점·그리기는 기존 `decide_grasp`/`draw_selection` 재사용.
  - `--image` 단일 이미지 모드(`run_single_image`): `run_realtime_seg.py`의
    같은 모드와 JSON 키(state/pose/reason/candidates) 동일.
  - CLI `--select`, `--selection-config`, `--human-prior-config`, `--image`,
    `--image-output`, `--hand-x`, `--hand-y`는 `run_realtime_seg.py`와 같은
    이름·기본값.
- 추가: `tests/test_rfdetr_selection_adapter.py` (테스트 8개)
- 수정 없음(계획서 "하지 말 것" 준수): `src/grasp_selection/*`,
  `scripts/run_realtime_seg.py`, `configs/grasp_selection.yaml`,
  `configs/human_grasp_prior.yaml`

### 검증 로그

1. 전체 테스트: `python -m unittest discover -s tests -p "test_*.py"` →
   **Ran 150 tests ... OK** (기존 142 + 신규 8). 신규 8개는 합성 3클래스 변환,
   base=1 정규화, None/빈 검출, 길이 불일치 오류, 후보 2개+회피 마스크 경로,
   저신뢰 0.25 필터링(손 오검출 회귀) 검증.
2. 단일 이미지 결정(mug_04, conf 0.25, 기본 손 위치=좌측 중앙, CPU):
   | 이미지(축약) | state | pose | 1위 후보 | 2위 후보 |
   |---|---|---|---|---|
   | 10_50_25 | GRASP | POWER | body 0.791 | handle 0.679 |
   | 10_51_20 | GRASP | POWER | body 0.791 | handle 0.672 |
   | 11_03_33 | GRASP | POWER | body 0.833 | handle 0.614 |
   손 위치를 손잡이 쪽으로 옮긴 추가 2건:
   | 11_03_33 `--hand-x 1100 --hand-y 280` | ALIGN | — | handle 0.752 | body 0.729 (점수 차 0.02 < 0.08) |
   | 10_51_20 `--hand-x 1300 --hand-y 230` | GRASP | WRAP | handle 0.852 | body 0.553 |
   손 적응형 순위 반전과 보수적 ALIGN 규칙이 설계대로 동작한다. 결정 JSON은
   `outputs/realtime_test/rfdetr_select_decision_*.json.log` 5건, overlay는
   `rfdetr_single_overlay_*.png`로 저장했다. 10_51_20 handside overlay에서
   선택 후보 파지점이 손잡이 마스크 안쪽(손 방향 쪽)에 찍히고 GRASP: WRAP
   배너·접근 화살표가 그려지는 것을 육안 확인했다.
3. 실시간 루프 headless 스모크: `--select --no-window --duration 10 --conf 0.25
   --device cpu` → 18프레임 무오류 완주(평균 490 ms/프레임, CPU 정상 범위).
   장면에 머그가 없어 매 프레임 NO_TARGET 경로와 functional 회피 마스크 경로가
   실행됐다.

### 참고 사항

- 폭 적합도는 mm 미보정으로 전 후보 0.5 중립이며 계획서 "함정" 절대로 정상이다.
  POWER/WRAP 구분은 body 클래스 규칙과 px 임시 기준으로 결정됐다.
- mug_04 이미지는 어댑터 동작의 정성 확인에만 썼다. 성능 수치 보고가 아니다.
- 남은 사용자 참여 항목: 실시간 창에서 마우스 모의 손 조작 확인(완료 기준 1의
  대화형 부분). headless 루프와 그리기 코드는 검증됐으므로 창 모드에서
  `python scripts/run_realtime_rfdetr.py --select --conf 0.25` 실행만 하면 된다.
