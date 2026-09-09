# Brunel Hand 제어 ↔ 비전 파이프라인 연동 계획

작성일: 2026-08-21
작성 당시 상태: 계획 단계 (펌웨어는 100 ms 펄스 수동 테스트, 위치 피드백 미해결)
2026-08-24 갱신: Phase 0·1 완료, Phase 2 펌웨어는 Jetson에 연결된 보드에 이미 올라가 있다.
남은 것은 Phase 2 벤치 시험(모터 구동)부터다. 9.4절 참고.

이 문서는 `firmware/hand_control/`의 Arduino 펌웨어를 폐루프 제어로 발전시키고,
`src/perception/realtime.py`의 실시간 후보 선택 결과(`Decision`)와 연결하기까지의
순서·인터페이스·안전 규칙을 정리한다. `PROJECT_CONTEXT.md` 12절(최소 명령
`OPEN/PRECISION/WRAP/POWER/STOP`, 상태 머신, 필수 안전 조건)과 `AGENTS.md` 8절
(짧은 펄스로 시작, 모든 구동 루프에 타임아웃과 `STOP`, 실측 전 값 확정 금지)을
그대로 따른다.

---

## 1. 현재 코드 상태 점검 (2026-08-21)

### 1.1 펌웨어 `firmware/hand_control/hand_control.ino` (초기 펄스 스케치, 2026-08-21 기준)

| 항목 | 상태 |
|---|---|
| 핀맵 | `MOTOR_IN1={2,4,6,8}`, `MOTOR_IN2={3,5,7,9}`, `FEEDBACK_PIN={A0..A3}`, 인덱스 0=검지, 1=중지, 2=약지·소지, 3=엄지 |
| 구동 방식 | 단일 문자 명령(`1`~`8`)당 100 ms 펄스 후 LOW/LOW(coast). `s` 전체 정지, `p` 위치 출력 |
| ADC | `analogReadResolution(12)`, 1회 샘플, 평균 없음 |
| 동작 확인 | 검지 양방향 구동 확인. 위치 피드백은 A0≈2~16으로 **미해결**(3V3 공급선 또는 핀5 배선 의심) |
| 한계 | `delay()` 기반이라 펄스 중 `s`를 못 받음. 최대 구동시간·스톨 감지·heartbeat 없음. 보정값 없음 |

버그는 없고 수동 테스트 용도로는 충분하다. 폐루프 버전은 별도 스케치로 새로 쓰되
기존 단일 문자 명령은 진단용으로 호환 유지한다.

### 1.2 비전·후보 선택 (`src/`)

| 모듈 | 상태 |
|---|---|
| `src/perception/realtime.py` | 카메라 루프 `run_camera_loop(model, config, selection_config, hand_tracker)`. 프레임마다 `extract_candidates → decide_grasp → draw_selection`. 종료/스냅숏 키 처리. **손 제어 훅은 아직 없음** |
| `src/grasp_selection/scoring.py` | `decide_grasp()` → `Decision(state ∈ {GRASP, ALIGN, NO_TARGET}, candidate, pose ∈ {PRECISION, WRAP, POWER}, reason, ranked)` |
| `src/grasp_selection/candidates.py` | 3클래스 모델(`handle_grasp_region`, `body_grasp_region`, `functional_region`) 기준 후보 추출 |
| `src/perception/hand_tracker.py` | 로봇손 위치·접근 방향 추정(`motion` 백엔드 기본). `HandState` 반환, 추적 끊기면 `None` |
| `configs/grasp_selection.yaml` | `pose.mm_per_px`, `max_grasp_width_mm` 등 **실측 대기(null)** — 손 최대 개구 폭 실측과 직접 연결됨 |
| 테스트 | `tests/test_grasp_selection.py` 12개, `tests/test_hand_tracker.py` 8개 통과 (`python -m unittest discover -s tests -p "test_*.py"`) |

### 1.3 아직 없는 것

- `src/hand_control/` 패키지(시리얼 클라이언트, 프리셋, 상태 머신) — `AGENTS.md` 권장 구조에만 존재
- 손 액추에이터 설정 파일(보정값·프리셋) — `configs/hardware/`에 없음
- `pyserial` 의존성(로컬 미설치, requirements에도 없음)
- 펌웨어 ↔ 호스트 시리얼 프로토콜 정의
- `CLAUDE_CODE_HANDOFF.md` 13.2의 "실시간 스크립트 없음" 서술은 코드보다 오래됨(이미 구현됨). 문서 갱신 대상

---

## 2. 목표 구조

```text
src/perception/realtime.py  ── Decision(state, pose, candidate), HandState 유효성 ──▶
src/hand_control/state_machine.py   SEARCH→TARGET_SELECTED→ALIGN→PRE_SHAPE→CLOSE→HOLD→OPEN
        │  프리셋 이름 + 손가락별 목표(ADC)            ▲ 펌웨어 이벤트(DONE/STALL/TIMEOUT)
        ▼                                              │
src/hand_control/client.py  ── 줄 단위 ASCII, 115200, heartbeat ──▶  firmware/hand_control (Nano 33 IoT)
        │                                                              │ 폐루프·타임아웃·스톨·STOP
configs/hardware/hand_actuators.yaml (보정값·프리셋, 단일 진실)       ▼
                                                            DRV8833 ×2 → PQ12 ×4
```

책임 분리:

- **펌웨어**: 저수준 안전(최대 구동시간, 스톨, heartbeat 끊김, `STOP` 최우선)과
  손가락별 위치 폐루프. 프리셋·정책을 모른다. 보정값은 컴파일 기본값 +
  호스트에서 `SET`으로 덮어쓰기(RAM) 가능.
- **호스트(Python)**: 보정값·프리셋의 단일 진실(YAML), 상태 머신, 비전 결과와의
  결합, 로그. 펌웨어를 다시 굽지 않고 YAML만 바꿔 튜닝한다.
- 프리셋은 손가락별 **닫힘 비율 0.0(완전 펼침)~1.0(보정된 접힘)** 로 정의하고
  호스트가 보정값으로 ADC 목표로 환산해 보낸다. 보정이 바뀌어도 프리셋은 그대로다.

---

## 3. 시리얼 프로토콜 v1 (제안)

- 115200 baud, 줄 단위 ASCII, LF 종료. 토큰은 영문(파싱 안정성), 설명은 주석으로.
- 기존 단일 문자 `1`~`8`, `s`, `p`는 진단용으로 유지(줄 끝 문자 무시).
- 모든 명령에 1줄 이상 응답: `OK ...` / `ERR <code> <설명>`. 비동기 이벤트는 `EVT ...`.

| 명령 | 응답 | 설명 |
|---|---|---|
| `PING` | `OK PONG <uptime_ms>` | 연결 확인, heartbeat로도 사용 |
| `STOP` | `OK STOP` | 전체 coast, 진행 중인 이동 취소. 어떤 상태에서도 최우선 |
| `POS` | `OK POS a0 a1 a2 a3` | N회 평균 ADC |
| `STATUS` | `OK STATUS <i>:<state>:<adc>:<target> ×4` | state ∈ IDLE/MOVING/DONE/STALL/TIMEOUT/UNCAL |
| `MOVE <i> <adc>` | `OK MOVE` 후 `EVT DONE <i>` 등 | 손가락 i를 ADC 목표로 폐루프 이동 |
| `MOVEALL <a0> <a1> <a2> <a3>` | 동일 | 네 손가락 동시 목표 (`-1`이면 유지) |
| `OPEN` | 동일 | 모든 손가락 보정된 OPEN 위치로 |
| `SET CAL <i> <open_adc> <closed_adc> <invert 0/1>` | `OK CAL` | 보정값 주입(RAM). 없으면 UNCAL로 구동 거부 |
| `GET CAL` | `OK CAL ...` | 현재 보정값 |
| `HELP` | 목록 | |
| 이벤트 | `EVT DONE <i>` / `EVT STALL <i> <adc>` / `EVT TIMEOUT <i> <adc>` / `EVT HEARTBEAT_STOP` | |

안전 매개변수(펌웨어 상수, `hand_config.h`):

- `MAX_MOVE_MS` 손가락별 1회 이동 상한(실측 전 stroke 시간을 모르므로 보수적으로 시작, 실측 후 조정)
- `STALL_WINDOW_MS` 동안 ADC 변화 < `STALL_MIN_DELTA` 이면 STALL 정지
- `DEADBAND_ADC` 목표 허용 오차, 도달 후 `REST_MS` 동안 재구동 금지(PQ12 duty 20 %)
- `HEARTBEAT_TIMEOUT_MS`: 이동 중 이 시간 동안 아무 명령도 없으면 전체 정지
- ADC는 8회 평균, 값 범위 밖(보정 범위 ± 여유)이면 센서 이상으로 정지

---

## 4. 단계별 계획

각 단계는 이전 단계의 **실측 결과**가 있어야 넘어간다. 모터가 도는 시험은
사용자가 배선·전원을 확인하고 현장에 있을 때만 한다.

### Phase 0 — 피드백 정상화 (완료, 2026-08-21 / 2026-08-24 재확인)

- 산출물: `firmware/hand_feedback_check/` (모터 구동 없음, A0~A3 원시값·평균·최소/최대 주기 출력)
- 절차: 10 V OFF·USB ON에서 3V3 → PQ12 핀4 → 핀5 → A0 전압 추적, wiper 핀 확정
  (분리 상태 저항 → 짧은 펄스 이동 → 재측정)
- 완료 기준: 네 채널 모두 위치에 따라 단조 변화, 정지 시 잡음 폭 기록

### Phase 1 — 보정값 실측 기록

- 손가락별 OPEN/CLOSED ADC, 방향(명령 1/2 ↔ 펼침/접힘), 전 행정 이동 시간(10 V), 정지 잡음 폭
- 산출물: `configs/hardware/hand_actuators.yaml` (측정 전 값은 `null`, 로더가 null이면 구동 거부)
- 함께 실측: 손 최대 개구 폭(mm) → `configs/grasp_selection.yaml`의 `max_grasp_width_mm` 갱신 근거

### Phase 2 — 폐루프 펌웨어

- **2026-08-21 작성·컴파일 완료: `firmware/hand_closed_loop/`** (`hand_closed_loop.ino`, `hand_config.h`,
  `hand_types.h`). 사용자의 펄스 스케치 `firmware/hand_control/`은 그대로 보존.
  `millis()` 기반 비차단 상태 머신, 3절 프로토콜(+ `FRAC`/`FRACALL` 닫힘 비율 명령, `PULSE`,
  `SET HOLD`), 안전 매개변수, `setup()`에서 모든 IN 핀 LOW, 단일 문자 명령 호환.
  벤치 시험은 아직 전(아래 완료 기준 확인 필요).
- 벤치 완료 기준: 시리얼 모니터로 `MOVE` 반복 시 목표 ± deadband 도달, 손으로 막았을 때
  `EVT STALL`, 끝 위치 근처에서 `EVT TIMEOUT`, 연속 명령 사이 휴지 준수, `STOP` 즉시 반응,
  heartbeat 끊김 시 정지

### Phase 3 — Python 클라이언트 `src/hand_control/`

| 파일 | 역할 |
|---|---|
| `protocol.py` | 명령 인코딩·응답/이벤트 파싱(입출력 없음, 단위 테스트 대상) |
| `presets.py` | YAML 로드, 닫힘 비율 → ADC 환산(invert 반영), null 검사 |
| `link.py` | `SerialLink`(pyserial, 타임아웃, 백그라운드 수신 스레드+큐) / `MockLink`(펌웨어 에뮬레이터) |
| `client.py` | `HandClient`: `connect()`(PING·SET CAL), `stop()`, `open()`, `move_preset(name)`, `wait_done(timeout)`, `status()`, 종료 시 `STOP` |
| 테스트 | `tests/test_hand_protocol.py`, `tests/test_hand_presets.py`, `tests/test_hand_client_mock.py` (unittest) |

- 의존성: `pyserial` 추가 위치 결정 필요(제안: Jetson 실행용 `requirements-runtime.txt` 신설)
- 카메라 루프를 막지 않도록 시리얼 수신은 스레드, 송신은 비차단

### Phase 4 — 세 프리셋 독립 시험 (`PROJECT_CONTEXT.md` 1주차 완료 기준)

- `scripts/run_hand_presets.py --port <PORT> --preset OPEN|PRECISION|WRAP|POWER --repeat N [--mock]`
- 프리셋 비율(손가락별 0~1)은 시행착오로 사용자가 정하고 YAML에 기록. 추측으로 채우지 않음
- 기록: 반복 성공률, 도달 시간, STALL/TIMEOUT 발생 → `outputs/hand_tests/*.jsonl`
- 확인: coast 상태에서 손가락이 위치를 유지하는지(아니면 brake 모드 검토)

### Phase 5 — 비전 통합

- `src/hand_control/state_machine.py`: `PROJECT_CONTEXT.md` 12절 상태 머신.
  입력 = 매 프레임 `Decision`, 손 추적 유효성, 키 입력; 출력 = `HandClient` 호출
  - `GRASP` 상태가 동일 후보·자세로 `stable_frames` 이상 유지 → `PRE_SHAPE`(자세별 반개방 프리셋)
  - `CLOSE`는 사용자 키(스페이스) 트리거를 기본으로, 자동 모드는 설정으로 분리
  - `HOLD` → `o` 키로 `OPEN`; `x` 키 = 비상정지(`STOP`)
  - `CLOSE` 전에 `ALIGN`/`NO_TARGET`/손 추적 소실 → 정지 후 `OPEN` 복귀; 파지 직전 대상 마스크 소실 → 정지
- `realtime.run_camera_loop(..., hand_controller=None)` 인자 추가(기본 None이면 기존 동작 불변),
  overlay에 손 상태·펌웨어 이벤트 표시
- `scripts/run_realtime_seg.py --hand-port <PORT> | --hand-mock`
- dry-run: `MockLink`로 카메라만 켜고 전체 흐름 검증 후 실제 포트 연결

### Phase 6 — 실제 파지 평가 (`PROJECT_CONTEXT.md` 13.3)

- Baseline 1(고정 POWER·지정 위치) / Baseline 2(최근접 후보·POWER) / Proposed
- 매 시도 JSONL 로그: 시각, decision, pose, 프리셋, 펌웨어 이벤트, 성공 여부(5 cm·3 s), 실패 원인

---

## 5. 설정 파일 초안 `configs/hardware/hand_actuators.yaml`

```yaml
schema_version: 1
serial:
  port: null            # 예: COM5 / /dev/ttyACM0 — 환경별, CLI가 우선
  baud: 115200
  heartbeat_ms: 300
fingers:                # 인덱스 순서 고정: 0 검지, 1 중지, 2 약지·소지, 3 엄지
  - name: index
    open_adc: null      # 실측 후 기입
    closed_adc: null    # 안전하게 접힌 위치(끝까지 아님)
    invert: null        # 명령 1이 펼침이면 false
    max_move_ms: null
  # ... middle, ring_little, thumb
presets:                # 손가락별 닫힘 비율 0.0~1.0, 시행착오 후 확정
  OPEN:      [0.0, 0.0, 0.0, 0.0]
  PRECISION: null
  WRAP:      null
  POWER:     null
  PRE_SHAPE: { PRECISION: null, WRAP: null, POWER: null }
```

---

## 6. 안전 체크리스트

펌웨어: 전원 인가·리셋 시 IN 핀 LOW / `STOP` 파싱은 다른 명령보다 먼저 / 손가락별
최대 구동시간 / 스톨 감지 / heartbeat 끊김 정지 / ADC 범위 이탈 정지 / 도달 후 휴지(duty 20 %) /
미보정(UNCAL) 상태 구동 거부.

호스트: 연결 시 보정값 주입 전 구동 금지 / 예외·종료 시 반드시 `STOP` / 손 추적·마스크
소실 시 정지 / 사용자 `x` 최우선 / 자동 CLOSE는 명시적 옵션 / 모든 시도 로그.

운용: 10 V는 사용자가 배선 확인 후 인가 / 업로드·진단은 10 V OFF에서 / 첫 시험은
짧은 펄스·한 손가락 / 실측 없는 값은 null로 두고 추측하지 않음.

---

## 7. 사용자 실측·결정이 필요한 항목

1. 피드백 정상화 결과(전압·ADC·wiper 핀) — Phase 0
2. 손가락별 OPEN/CLOSED ADC, 방향, 전 행정 시간 — Phase 1
3. 손 최대 개구 폭(mm), `T_precision`/`T_wrap` — `grasp_selection.yaml`
4. 프리셋별 손가락 닫힘 비율 — Phase 4 시행착오
5. Arduino 시리얼 포트 이름(Windows/Jetson), `pyserial` 의존성 추가 위치
6. 자동 CLOSE 허용 여부(기본: 사용자 키 트리거)

## 8. 문서 갱신 예정

- `PROJECT_CONTEXT.md` 12절에 프로토콜 문서 링크, `AGENTS.md` 3절 구조에 `firmware/` 추가(사용자 승인 후)
- `CLAUDE_CODE_HANDOFF.md` 13.2 "실시간 스크립트 없음" 갱신

---

## 9. 실측 기록

값은 사용자가 시리얼 모니터에서 읽은 그대로 적는다. 1샘플 `analogRead` 기준이며
반복 측정 전까지 **잠정값**이다. 보정 YAML에는 재현 확인 후 기입한다.

### 9.1 2026-08-21 검지(A0) 첫 피드백 확인 — 100 ms 펄스 스케치

전제: 시작 시 검지는 펼친 상태(사용자 설명). 10 V ON, 배선 재연결 후.

| 순서 | 명령 | A0 | 변화 | 해석 |
|---|---|---|---|---|
| 1 | 방향 1 | 3452 | — | 펼침 한계에서 스톨(움직임 없음) |
| 2 | 방향 1 | 3457 | +5 | 〃 (잡음 수준) |
| 3 | 방향 2 | 3273 | −184 | 기동 구간(정지 마찰·백래시) |
| 4 | 방향 2 | 2784 | −489 | 정속 구간 |
| 5 | 방향 2 | 2384 | −400 | 정속 구간 |
| 6 | 방향 2 | (기록 누락) | | 접힘 끝 값 미확인 |

잠정 결론:

- 피드백 정상 동작 확인(이전 2~16 문제는 배선 재연결로 해소).
- 방향 1 = 펼침(ADC 증가), 방향 2 = 접힘(ADC 감소). 육안 재확인 필요.
- 검지 펼침 한계 ≈ 3450 (끝 위치 스톨 값). OPEN 목표는 이보다 50~100 안쪽으로 둘 것.
- 정속 이동 속도 ≈ 4~5 count/ms (100 ms 펄스당 400~490). 폐루프는 100 ms 펄스가 아니라
  연속 구동 + 5~10 ms 주기 ADC 확인으로 설계해야 하며 deadband 30~50 필요.
- 정지 잡음 ±5 수준(1샘플).
- 미측정: 접힘 한계 값, 전 행정 시간, 반복 재현성, A1~A3 값.

### 9.2 2026-08-21 17:48 네 손가락 펄스 로그 — `outputs/hand_tests/serial_20260821_174804.log`

`scripts/hand_serial_terminal.py`로 기록. 100 ms 펄스를 한 번씩 보내며 각 펄스 직후 1샘플.

**검지(A0) — 정상, 전 행정 왕복 확인**

| 구간 | 값 | 펄스당 변화 |
|---|---|---|
| 시작(펼침 한계에서 방향 1) | 3436 | 스톨 |
| 방향 2 접힘 ×10 | 3251 → 2857 → 2437 → 1968 → 1523 → 1144 → 857 → 608 → 381 → 365 | −185, −394, −420, −469, −445, −379, −287, −249, −227, **−16(한계)** |
| 방향 1 펼침 ×14 (365에서) | 570 → 942 → 1262 → 1573 → 1826 → 2041 → 2282 → 2485 → 2679 → 2888 → 3095 → 3241 → 3305 → 3426 | +205, +372, +320, +311, +253, +215, +241, +203, +194, +209, +207, +146, +64, +121 |
| 정지 후 `s` 출력 | 3423 | 잡음 −3 |

- 검지 **접힘 한계 ≈ 365~380**, **펼침 한계 ≈ 3425~3457**(4회 관측: 3426, 3436, 3452, 3457 — 끝 위치 값이 ±15 정도 흔들림). 가용 범위 ≈ 3060 count.
- 접힘(방향 2)이 펼침(방향 1)보다 빠르다: 중간 구간 접힘 약 4~4.7 count/ms, 펼침 약 2~3.7 count/ms.
  양 끝 근처에서 속도가 떨어진다(부하 증가). 폐루프는 방향별 속도 차이와 끝 근처 감속을 감안해야 한다.
- 전 행정 구동 시간: 접힘 약 1.0 s(10펄스), 펼침 약 1.4 s(14펄스) → `MAX_MOVE_MS` 초기값 2000~2500 ms 권장(실측 기반).
- 제안 소프트웨어 한계(사용자 확인 필요): `OPEN_POS ≈ 3350`, `CLOSED_POS ≈ 450` (기계 한계에서 80~100 count 안쪽).

**중지(A1), 약지·소지(A2), 엄지(A3) — 피드백 미동작**

| 채널 | 펄스 수 | 값 범위 | 판정 |
|---|---|---|---|
| A1 중지 | 25 (방향 1 ×10, 방향 2 ×15) | 0~11 | 위치와 무관한 0 근처 → 이전 A0와 같은 증상(3V3 또는 wiper 배선) |
| A2 약지·소지 | 22 (방향 2 ×9, 방향 1 ×13) | 0~64 | 〃 (간헐적 64·40은 떠 있는 입력의 잡음으로 보임) |
| A3 엄지 | 40 (방향 1 ×27, 방향 2 ×13) | 0~20 | 〃 |

- 모터가 실제로 움직였는지는 로그로 알 수 없음(사용자 육안 확인 필요).
- 세 채널이 동시에 같은 증상이므로 3V3 분배(핀 4)의 공통 원인 가능성이 높다. 검지를 고칠 때 검지 핀 4만
  3V3에 직접 다시 연결했다면 나머지 세 개의 3V3 경로를 우선 점검한다.
- 후속: 사용자가 모터 4개 모두 정상 구동 확인. 21:14경 배선 수정으로 A1~A3 정상화
  (`serial_20260821_210234.log`에서 세 채널이 동시에 살아남 → 3V3 분배가 원인이었던 것으로 보임).

### 9.3 2026-08-21 21:50 네 손가락 왕복 로그 — `serial_20260821_215011.log` (보정 기준)

100 ms 펄스를 한 번씩, 값이 멈출 때까지 한 방향 → 반대 방향. 기계 한계는 1샘플 관측값.

| 손가락 | ADC 증가 방향 | 낮은 한계 | 높은 한계 | 100 ms당 이동(ADC↓ / ADC↑) | 전 행정 시간(↓ / ↑) |
|---|---|---|---|---|---|
| 검지 A0 | 방향 1 | 333 (관측 333~380) | 3446~3457 (관측 3409~3457) | 400~480 / 200~300 | 1.0 s / 1.5 s |
| 중지 A1 | **방향 2** (검지와 반대) | 432 (−84 진행 중) | 3495~3500 | 350~480 / 180~300 | 1.0 s / 1.4 s |
| 약지·소지 A2 | 방향 1 | 422 (−88 진행 중) | 3522~3550 | 380~470 / 150~350 | 0.9 s / 1.2 s+ |
| 엄지 A3 | 방향 1 | 309 (관측 309~336) | 3552 | 160~280 / 120~200 | 1.6 s / 2.2 s |

관찰:

- 네 손가락 모두 **ADC가 낮아지는 방향이 더 빠르다**(부하 비대칭). 검지에서 낮은 ADC = 접힘이
  확인됐으므로 나머지도 낮은 ADC = 접힘일 가능성이 높지만, 중지·약지·소지·엄지는 육안 확인 전.
- 중지는 방향 번호와 ADC 증감 대응이 검지와 반대 → 모터선 또는 BIN1/BIN2 배선이 뒤집힌 것. 배선을
  바꾸지 않고 펌웨어 설정(`adc_increase_direction`)으로 흡수한다.
- 끝 위치에서 같은 방향을 한 번 더 주면 값이 약 40 되돌아오는 현상(3446→3409, 3497→3454) → 기계 한계
  값은 ±40 흔들림. 소프트 한계는 80 이상 안쪽으로.
- 엄지는 다른 손가락보다 느리다(ADC↑ 약 1.5~2 count/ms).
- 정지 잡음(진단 스케치, 8회 평균, 20 s): 범위 28~58 count, 표준편차 5~10 → deadband 50, 스톨 판정
  200 ms에 60 미만 이동.
- 세션 종료 시 정지값: A0 3435, A1 3421, A2 3509, A3 336.

이 값으로 `configs/hardware/hand_actuators.yaml`을 작성했다.

**2026-08-21 21:55 사용자 확인**: 네 손가락 모두 "펼침 → 접힘 → 펼침" 왕복을 마친 펼친 상태였다
(A0 3435, A1 3421, A2 3509, A3 336). 따라서 검지·중지·약지·소지는 **높은 ADC = 펼침**, 엄지는
**낮은 ADC = 펼침**(센서 극성 반대) → `extended_is_high` 확정, `calibration_confirmed: true`.
펼침 명령: 검지 방향 1, 중지 방향 2, 약지·소지 방향 1, 엄지 방향 2.

### 9.4 2026-08-24 Jetson 연결 후 피드백 재확인 (모터 구동 없음)

보드를 Jetson USB로 옮겨 연결했다. `/dev/ttyACM0`으로 잡히며 장치는
`2341:8057 Arduino SA Arduino NANO 33 IoT`다. `sail-student`가 `dialout` 그룹에
없어 포트가 열리지 않았고, 사용자 승인 아래 `usermod -aG dialout sail-student`로
해결했다. 읽기 전용 확인이므로 모터를 구동하는 명령은 보내지 않았다.

보드에는 이미 폐루프 펌웨어가 올라가 있다. 응답 형식이 `OK POS ...`이고
`GET CAL`, `GET PARAM`, `STATUS`가 모두 동작한다.

`GET CAL` 응답: `0:450:3330:1:1:3000 1:520:3400:2:1:3000 2:520:3420:1:1:3000 3:420:3450:1:0:3500`.
`configs/hardware/hand_actuators.yaml`의 방향(검지 1, 중지 2, 약지·소지 1, 엄지 1)과
`extended_is_high`(검지·중지·약지·소지 참, 엄지 거짓)이 그대로 반영돼 있다.

`GET PARAM` 응답: `deadband=50 hb=0 hold=0 stall_window=200 stall_delta=60 rest=500 loop=5`.

`STATUS`는 네 채널 모두 `IDLE`이었다. 정지 상태에서 `p`를 60회 보내 모은 값이다.

| 손가락 | 표본 | 최소 | 최대 | 폭 | 평균 | 표준편차 |
|---|---|---|---|---|---|---|
| 검지 | 60 | 3293 | 3327 | 34 | 3311.7 | 6.90 |
| 중지 | 60 | 3358 | 3402 | 44 | 3384.7 | 9.28 |
| 약지·소지 | 60 | 3393 | 3433 | 40 | 3411.3 | 8.28 |
| 엄지 | 60 | 384 | 415 | 31 | 401.4 | 5.63 |

네 채널 모두 값을 정상으로 내보내고, 위치는 보정 기준으로 네 손가락 다 펼침 쪽이다.
잡음 폭 31~44는 9.3절에서 기록한 28~58 범위 안이라 재현된다. `deadband=50`이
최대 잡음 폭 44보다 크므로 폐루프가 잡음만으로 진동하지는 않는다. 다만 중지의
여유가 6 count뿐이라 온도나 배선 상태가 바뀌면 다시 확인해야 한다.

엄지는 평균 401로 보정 하한 420보다 19 count 낮다. 잡음 폭 31의 절반 수준이라
당장 문제는 아니지만, 펼침 목표를 420으로 잡는 `FRAC 0` 명령이 이미 지나친 위치를
목표로 삼게 되므로 Phase 2 벤치 시험에서 실제 거동을 확인해야 한다.

Phase 0 완료 기준 두 가지 중 "정지 시 잡음 폭 기록"은 위 표로 충족했다.
"네 채널 모두 위치에 따라 단조 변화"는 모터를 움직여야 하므로 확인하지 않았다.
사용자가 배선과 전원을 확인하고 현장에 있을 때 Phase 2 벤치 시험에서 함께 본다.

사용한 스크립트는 Jetson `~/hjh_vision_hand/scripts/read_hand_feedback.py`와
`sample_hand_rest.py`다. 둘 다 모터를 구동하는 명령을 거부하도록 허용 명령 목록을
`p`, `h`, `STATUS`, `GET CAL`, `GET PARAM`으로 제한했다.

### 9.5 2026-08-24 Phase 2 벤치 시험 1차 — 정지 방식이 정지 오차를 지배한다

사용자가 10 V 모터 전원을 연결한 뒤 검지(인덱스 0)로만 시험했다. `AGENTS.md`의
"짧은 펄스로 시작" 규칙에 따라 100 ms 펄스부터 올렸고, 매 시행 전후와 예외 발생 시
`STOP`을 보냈다. 네 손가락이 모두 펼침 끝단 근처였으므로 기계 한계로 미는 것을 피해
접힘 방향으로만 움직였다.

#### 펄스 길이별 이동량

| 펄스 | 총 이동 | `EVT PULSE_DONE` 보고 이동 | 차이 |
|---|---|---|---|
| 50 ms | 272 | 61 | 211 |
| 100 ms | 399 | 170 | 229 |
| 200 ms | 610 | 336 | 274 |

명령 시간에 비례하는 성분은 약 2.1~2.5 count/ms이고, 여기에 펄스 길이와 무관한
150~190 count의 고정 성분이 더 붙는다. 이 고정 성분이 구동을 끊은 뒤의 관성 이동이다.
`EVT PULSE_DONE`이 보고하는 값은 `coastFinger()` 직후 `readAdc()`로 읽은 것이라
아직 멈추지 않은 시점의 위치다. 펌웨어 결함은 아니지만 이 값을 정지 위치로 믿으면 안 된다.

관성 이동은 구동 차단 후 230 ms 안에 끝난다. 그 뒤 값은 잡음 폭(±12) 안에서만 흔들린다.
처음에는 0.6 s 뒤 한 번만 읽고 관성이 없다고 판단했는데, 표본이 늦어 이미 끝난 뒤를
본 것이었다. 펄스 길이를 바꿔 세 점을 재고서야 고정 성분이 드러났다.

#### 폐루프 `MOVE` 정지 오차: `SET HOLD` 0 대 1

목표 1640과 2140을 3회씩 왕복했다. 보정 끝단에서 300 안쪽으로만 목표를 잡았다.

| `hold` | 정지 방식 | 절대오차 평균 | 절대오차 최대 | deadband 50 기준 |
|---|---|---|---|---|
| 0 | coast | 84.7 | 197 | 초과 |
| 1 | brake 유지 | 17.2 | 34 | 통과 |

`hold=0`에서는 아래로 내려가는 이동의 오차가 −197, −178로 특히 컸고 올라가는 이동은
+34~+53이었다. 방향에 따라 오차가 비대칭이다. `hold=1`에서는 양방향 모두 ±34 안에
들어온다.

`finishMove()`는 `holdBrake`가 참이고 상태가 `ST_DONE`이나 `ST_STALL`일 때만
`brakeFinger()`를 부른다. 현재 `HOLD_BRAKE_DEFAULT`가 0이라 기본값으로는 coast로
멈추고, 그 결과 목표를 최대 197 count 지나친다. `PULSE`는 `holdBrake`와 무관하게
항상 coast로 끝난다.

**결론: 폐루프 제어에는 `SET HOLD 1`이 필수다.** deadband를 관성 이동보다 크게
키우는 방법도 있으나, 그러면 정지 정밀도가 ±200 count가 되어 전체 행정 약 2,900의
7 %에 이르므로 프리셋 구분에 쓸 수 없다. `HOLD_BRAKE_DEFAULT`를 1로 바꾸거나,
Phase 3의 `HandClient.connect()`가 `SET CAL`과 함께 `SET HOLD 1`을 보내야 한다.
어느 쪽으로 할지는 사용자 결정 사항이다.

#### 아직 확인하지 않은 완료 기준

`STOP` 즉시 반응, 연속 명령 사이 휴지 준수, heartbeat 끊김 시 정지는 자동으로 시험할
수 있으나 이번 회차에서 하지 않았다. 손으로 막았을 때 `EVT STALL`은 사용자의 손이
필요하고, 끝 위치 근처 `EVT TIMEOUT`은 기계 한계로 미는 동작이라 사용자가 지켜보는
자리에서만 한다. 검지 외 세 손가락은 아직 구동하지 않았다.

사용한 스크립트는 Jetson `~/hjh_vision_hand/scripts/`의 `bench_pulse.py`,
`bench_coast.py`, `bench_move.py`다.

### 9.6 2026-08-24 폐루프 실동작 시험 (10 V 전원 인가, Windows COM10)

펌웨어 `firmware/hand_closed_loop` v2를 업로드하고 `scripts/hand_cli.py`·직접 시리얼로 확인했다.

**단일 손가락 30 % 접기 → 펼치기 (deadband 50)**

| 손가락 | 목표 | 도달 | 오차 | 시간 | 결과 |
|---|---|---|---|---|---|
| 검지 | 2466 / 3330 | 2501 / 3285 | +35 / −45 | 453 / 656 ms | DONE |
| 중지 | 2536 / 3400 | 2586 / 3353 | +50 / −47 | 485 / 687 ms | DONE |
| 약지·소지 | 2550 / 3420 | 2600 / 3380 | +50 / −40 | 265 / 703 ms | DONE |
| 엄지 | 1329 / 420 | 1285 / 470 | −44 / +50 | 625 / 547 ms | DONE |

**전체 접기/펼치기 (네 손가락 동시)**: 모두 DONE. 접기 1.59~2.16 s, 펼치기 1.74~1.86 s.
엄지가 가장 느리다(실측 예상과 일치).

**deadband 조정 (검지 2800↔3200 왕복 3회)**

| deadband | 평균 \|오차\| | 도달 시간 평균 | 결과 |
|---|---|---|---|
| 50 | 45.5 | 305 ms | DONE (진동 없음) |
| 25 | 20.5 | 323 ms | DONE |
| 12 | 7.0 | 349 ms | DONE |

deadband를 줄여도 진동이나 재구동이 없었다. 도달 오차는 항상 deadband 경계 근처이며,
제어 주기 5 ms당 약 20 count 이동하므로 정지 granularity가 그만큼이다. **기본값을 20으로 바꿨다.**

**정지 후 위치 유지 (검지, 목표 2800)**

| 설정 | 도달 직후 | 3초 후 | 변화 |
|---|---|---|---|
| `SET HOLD 0` (coast) | 2804 | 2525 | **−279** |
| `SET HOLD 1` (brake) | 2823 | 2783 | −40 |

coast로는 손가락이 자체 하중으로 처진다. 파지 유지에 필수이므로 **`HOLD_BRAKE_DEFAULT`를
true로 바꿨다**(YAML `control.hold_brake: true`, 클라이언트가 연결 시 `SET HOLD 1` 주입).

**기본값 변경 후 재확인**: `OK PARAM deadband=20 hb=900 hold=1 legacy=0`.
전체 접기 오차 +12/+11/+13/−17, 펼치기 −10/−2/−17/+11 (이전 ±40~50에서 개선).

**전원 없을 때(2026-08-24 오전) 확인한 안전 동작**: 네 손가락 모두 200 ms 안에 `EVT STALL`,
`MOVE 0 4000` → `ERR RANGE index`, `4` → `ERR LEGACY_OFF`, `s` → `OK STOP`.

**막힘(스톨) 시험 — 사용자가 검지를 손으로 막은 상태**

```
CLOSE 0 (목표 450, 시작 3336)
  +2297 ms  EVT STALL 0 1000      막힌 지점에서 정지
  유지 3초   1008 → 1030          brake로 +22 count만 밀림 (2초 이후 고정)
OPEN 0
  +1375 ms  EVT DONE 0 3312       정상 복귀
```

파지 동작 전 과정이 확인됐다: 저항을 만나면 목표까지 밀어붙이지 않고 정지하고, brake로
쥔 위치를 유지하고, 명령으로 풀린다. 막힌 지점 1008은 검지 기준 81 % 접힘이다.
같은 방식으로 실제 물체(머그)를 쥐면 손가락별 접촉 위치가 기록되므로 그 값이
`WRAP`/`POWER` 프리셋 비율의 실측 근거가 된다.

참고: 손으로 가볍게 누르는 정도(첫 시도)로는 스톨이 걸리지 않고 목표까지 갔다
(1812 ms, 자유 상태 1594 ms). 스톨 판정이 "200 ms 동안 60 count 미만 진행"이고 자유 속도가
200 ms에 약 800 count이므로, 거의 완전히 멈춰야 스톨로 본다. 소프트 한계 근처에서
액추에이터가 자연히 느려지는 것(100 ms에 20~60 count)을 오판하지 않으려면 이 정도 여유가 필요하다.

**아직 못 한 것**: 실제 물체(머그컵)로 네 손가락 동시 파지,
`PRECISION`/`WRAP`/`POWER` 프리셋 비율 결정.

### 9.7 2026-08-24 젯슨에서 머그컵 파지 — 구동 순서가 성패를 갈랐다

손 USB를 젯슨(`/dev/ttyACM0`)으로 옮기고 `src/hand_control` + `scripts/hand_cli.py`로 시험했다.

**같은 컵·같은 위치에서 연속 비교**

| 방식 | 결과 |
|---|---|
| 네 손가락 동시(`FRACALL` 한 번) | 검지 452·중지 520·약지·소지 530·엄지 3431 — **전부 100 % 닫힘**(컵이 밀려나 헛닫힘) |
| 엄지 먼저 → 나머지 세 개 | 엄지 DONE 3455(100 %) 후 검지 STALL 978(82 %)·중지 1147(78 %)·약지·소지 1131(79 %) — **실제 파지** |

엄지가 받침이 되기 전에 손가락이 도착하면 물체가 손가락 경로에서 밀려난다. 그래서
프리셋에 **구동 순서**를 도입했다(`configs/hardware/hand_actuators.yaml`의 `preset_sequence`).
각 프리셋의 순서는 네 손가락을 정확히 한 번씩 포함해야 하고, 로더가 이를 검사한다.

```yaml
preset_sequence:
  OPEN:      [[0, 1, 2, 3]]
  PRECISION: [[2, 3], [0, 1]]
  WRAP:      [[3], [0, 1, 2]]
  POWER:     [[3], [0, 1, 2]]
```

구현: `HandCalibration.preset_steps()`, `HandClient.move_preset_step()`/`run_preset()`,
`GraspController`가 CLOSE 상태에서 단계를 순서대로 진행한다(비차단).

**정식 경로(CLI)로 재현**: `hand_cli.py preset POWER` →
엄지 DONE 3434(100 %), 검지 STALL 947(82 %), 중지 STALL 1040(81 %), 약지·소지 STALL 1032(82 %).
수동 시험의 78~82 %와 일치한다. 이 값이 이 머그컵 몸통 지름에 해당하는 접힘 비율이다.

**부수 발견 — 다른 프로세스와의 포트 충돌**: 첫 파지 시도에서 응답이 한 칸씩 밀리고
보내지 않은 `OK MOVE 0 3300`이 섞여 들어왔다. 다른 프로세스가 같은 포트에 명령을 보내면
`exclusive=True`로도 막히지 않는다(상대가 flock을 걸지 않으면). `HandClient._request`가
명령별 기대 응답 종류를 확인하고 어긋난 `OK` 줄은 버리도록 고쳤고, 연결 시 조용해질 때까지
반복해서 수신 버퍼를 비우게 했다.

**아직 못 한 것**: 머그 손잡이로 `WRAP` 시험, `PRECISION`(펜 등) 시험, 들어올려 유지하는 시험,
`PRE_SHAPE` 값 결정(현재 전부 0 = 완전 펼침).

### 9.8 2026-08-24 세 프리셋 자세 검증과 파지 성공 판정에 대한 정정

젯슨에서 세 프리셋을 모두 구동해 자세 형성을 확인했다.

| 프리셋 | 결과 | 관측 |
|---|---|---|
| `POWER` | 머그컵 파지 성공 | 엄지 100 % 받침 후 세 손가락 78~82 %에서 `STALL` |
| `WRAP` | 자세 정확, 손잡이 파지 성공 | 엄지 −1 %(펼침 유지), 세 손가락 99~100 % `DONE` |
| `PRECISION` | 자세 정확, 파지 성공 | 1단계 약지·소지 0 %·엄지 100 %, 2단계 검지·중지 99~100 % `DONE` |

**정정: `STALL`만으로 파지 성공을 판단할 수 없다.** 처음에는 "STALL = 물체를 쥠,
DONE = 놓침"으로 해석했으나 사용자 확인 결과 다음이 맞다.

- `WRAP`은 손가락이 **구조상 그 이상 굽지 않는** 지점에서 끝나므로 손잡이를 걸어도 `DONE`이다.
- `PRECISION`은 얇은 물체를 손가락이 **끝까지 닫히면서** 쥐므로 `DONE`이다.
- `STALL`은 머그 몸통처럼 **두꺼운 물체**에 걸렸을 때 나타난다.

따라서 `MoveResult.grasped`를 `MoveResult.blocked`로 바꾸고(막혀서 멈췄다는 사실만 뜻함),
상태 머신의 표시도 "막힘 [...]" / "목표 도달"로 사실만 적게 했다. 파지 성공 판정은
`PROJECT_CONTEXT.md` 13.3의 기준(5 cm 들어올려 3초 유지)처럼 별도 확인이 필요하다.

9.7절의 머그 `POWER` A/B 비교는 그대로 유효하다. 머그 몸통은 두꺼워서, 동시에 닫아 100 %까지
간 것은 물체가 손가락 사이에 없었다는 뜻이 맞다. 다만 그 해석을 얇은 물체에 그대로 적용하면 안 된다.

### 9.9 2026-08-24 Phase 5 — RF-DETR 경로에 손 제어 연결 (mock 검증)

배포 모델 M의 실시간 경로 `scripts/run_realtime_rfdetr.py`에 `run_realtime_seg.py`와
같은 손 제어 연결(`--hand-port`/`--hand-mock`/`--hand-config`/`--hand-auto-close`/
`--hand-stable-frames`)을 추가했다. `HandClient` 연결·`GraspController` 프레임별
갱신·`draw_hand_status` 표시·키(스페이스/o/x)·종료 시 비상정지→shutdown 순서 모두
seg 경로를 미러링했고, 손 제어는 `--select` 없이는 거부한다(exit 2).

검증:

1. 전체 테스트 `python -m unittest discover -s tests -p "test_*.py"` → 161개 OK.
2. 로컬 CPU mock: `--select --hand-mock --no-window --duration 12` → 42프레임 완주,
   MockLink 연결 로그에 BOOT→ID→STOP→SET LEGACY 0→SET HB 900 순서 확인,
   요약에 `hand_state`/`hand_detail` 기록, 종료 시 비상정지.
3. 젯슨 GPU mock: 같은 명령 `--device cuda:0 --duration 15` → 80프레임 완주
   (중앙값 129 ms), 장면의 머그가 body 67·handle 16회 검출. 젯슨의
   `src/grasp_selection`이 `extract_candidates_from_arrays` 이전 구버전이라
   ImportError가 났고, 로컬 최신본으로 동기화해 해결했다.

mock은 배선·수명주기 검증이고 상태 전이 자체는 `tests/test_hand_state_machine.py`가
따로 검증한다. 실제 손으로 CLOSE까지 가는 통합 시험은 사용자가 현장에서 창 모드로
실행해야 한다(마우스 모의 손 + 스페이스 트리거):

```bash
# 젯슨 데스크톱에서 (모니터 필요)
.venv/bin/python scripts/run_realtime_rfdetr.py \
  --model outputs/training/custom_finetune_m_rfdetr_customv3_seed42/checkpoint_best_ema.pth \
  --select --hand-port /dev/ttyACM0 --conf 0.25 --device cuda:0
```

남은 것: 실기 통합 시험(위 명령), 파지 성공 판정(5 cm 들어올려 3초 유지),
PRE_SHAPE 값, ArUco `mm_per_px` 실측. 시리얼 포트는 한 프로세스만 열어야 한다
(9.7절 포트 충돌 — 당시 섞여 들어온 `MOVE 0 3300`은 다른 Claude 세션이 검지 복귀를
시도하며 보낸 명령이었음을 확인했다).
