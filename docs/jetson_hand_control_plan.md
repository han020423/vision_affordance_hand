# 젯슨에서 Brunel Hand를 제어하기 위한 계획

작성일: 2026-08-24
대상: Jetson Orin Nano Super (`ssh jetson`, 작업 디렉터리 `~/hjh_vision_hand`)
관련 문서: `docs/hand_control_integration_plan.md` (전체 연동 계획·실측 기록),
`PROJECT_CONTEXT.md` 12절 (손 제어 명령·상태 머신·안전 조건)

이 문서는 **젯슨이 USB 시리얼로 Brunel Hand를 구동**하게 만들기까지 필요한 작업만 다룬다.
비전(분할·후보 선택) 결과와의 결합은 마지막 단계에서 붙이고, 그 앞 단계까지는
비전 없이 손만 독립적으로 시험한다(`AGENTS.md` 8절: 카메라/모델 코드와 Arduino 코드를
독립 시험한 후 통합).

---

## 1. 확인된 현재 상태

### 1.1 손 쪽 (2026-08-21 실측 완료)

- 네 손가락 모터·위치 피드백 모두 정상. 기계 한계와 방향은
  `configs/hardware/hand_actuators.yaml`에 기록(`calibration_confirmed: true`).
- 폐루프 펌웨어 `firmware/hand_closed_loop/` v2를 2026-08-24에 보드에 업로드하고 프로토콜을
  실물 검증했다(`ID`, 보정 주입·검증, 안전 거부, 스톨 감지). **모터 전원(10 V)이 꺼져 있어
  실제 이동은 아직 확인하지 못했다.**
- 프리셋(`PRECISION`/`WRAP`/`POWER`)의 손가락별 닫힘 비율은 미정(사용자 시행착오 필요).

### 1.2 젯슨 쪽 (2026-08-24 원격 확인)

| 항목 | 확인 결과 |
|---|---|
| 접속 | `ssh jetson` 키 인증 정상, aarch64 |
| 작업 디렉터리 | `~/hjh_vision_hand` (안에 `src/`, `scripts/`, `outputs/`, 로그들) |
| 파이썬 | 시스템 `python3` 3.12.3, 가상환경 `~/hjh_vision_hand/.venv` |
| pyserial | **시스템·venv 모두 3.5 설치됨** (추가 설치 불필요) |
| 시리얼 권한 | 사용자가 이미 `dialout` 그룹에 속함 (추가 설정 불필요) |
| 현재 tty | `/dev/ttyACM*` 없음 — 손이 Windows(COM10)에 물려 있어서 정상 |
| ModemManager | **active** — 아래 4.2의 조치 필요 |
| `configs/` | **젯슨에 없음** — 보정값 YAML을 옮겨야 한다 |

---

## 2. 역할 분담

```text
[젯슨]  비전 → Decision(pose)                 [Nano 33 IoT]
        src/hand_control/state_machine.py
                 │ 프리셋 이름
                 ▼
        src/hand_control/client.py  ── USB CDC 115200, 줄 단위 ASCII ──▶ firmware/hand_closed_loop
                 │  ▲                                                      │ 위치 폐루프
        configs/hardware/hand_actuators.yaml                               │ 타임아웃·스톨·센서이상
        (보정값·프리셋의 단일 진실)  ◀── EVT DONE/STALL/TIMEOUT ───────────┘ STOP 최우선
                                                                    DRV8833 ×2 → PQ12 ×4
```

원칙:

- **저수준 안전은 펌웨어가 최종 책임**을 진다. 젯슨이 죽거나 USB가 빠져도 펌웨어가 스스로 멈춘다.
- **보정값·프리셋은 젯슨 YAML이 단일 진실**이다. 펌웨어를 다시 굽지 않고 YAML만 바꿔 조정한다.
  젯슨은 연결 시 `SET CAL`로 값을 주입한다.
- 프리셋은 손가락별 **닫힘 비율 0.0~1.0**으로 표현한다. 보정값이 바뀌어도 프리셋 정의는 그대로다.

---

## 3. 펌웨어 쪽 작업

### 3.1 v2에서 이미 반영한 젯슨 대비 사항

| 항목 | 내용 |
|---|---|
| `ID` 명령 | `OK ID hand_closed_loop fw 2 proto 1 fingers 4 adcbits 12 uptime <ms>`. 호스트가 연결 직후 펌웨어를 검증한다. **옛 펄스 스케치는 `ID`에 응답하지 않으므로 잘못된 펌웨어를 즉시 걸러낸다** (2026-08-21에 실제로 옛 스케치가 남아 `moveall`이 펄스로 해석된 사고가 있었다) |
| `SET LEGACY 0` | 단일 문자 명령(`1`~`8`, `p`, `h`)을 끈다. 헤드리스에서 잡음 한 글자로 모터가 도는 것을 막는다. `s`(정지)는 꺼도 항상 받는다 |
| 문서화된 USB CDC 특성 | 네이티브 USB라 포트를 열어도 보드가 리셋되지 않음 → `EVT BOOT`를 기다리지 말고 `ID`로 확인. `if (Serial)`은 SAMD 코어 구현에 `delay(10)`이 있어 제어 루프에서 쓰지 않음 |
| 출력 지연 한계 | 호스트가 읽지 않으면 USB 전송이 최대 70 ms(`TX_TIMEOUT_MS`) 지연될 수 있으나, 이벤트 출력은 **모터를 정지시킨 뒤** 수행하므로 지연 중 구동이 이어지지 않음 |

### 3.2 젯슨 운용 시 호스트가 연결 직후 보낼 것

```text
ID                → 펌웨어 확인 (응답 없거나 이름/proto 불일치면 즉시 중단)
STOP              → 알 수 없는 이전 상태 정리
SET LEGACY 0      → 단일 문자 명령 차단
SET HB 1000       → heartbeat 켜기 (이동 중 1 s 침묵이면 펌웨어가 전체 정지)
SET CAL 0..3 ...  → YAML 보정값 주입
GET CAL / GET PARAM → 주입 결과 확인
POS               → 초기 위치 기록
```

### 3.3 heartbeat가 곧 사망 감지

USB 분리·프로세스 강제 종료·SSH 끊김을 따로 감지하지 않는다. 어느 경우든 **명령이 끊기므로**
이동 중이면 `SET HB`로 정한 시간 안에 펌웨어가 전체 정지하고 `EVT HEARTBEAT_STOP`을 남긴다.
heartbeat를 끈 상태(`SET HB 0`)에서 호스트가 죽으면 손가락별 `max_move_ms`(3~3.5 s) 안에서만
구동이 이어진다 — 그래도 무한 구동은 없다.

### 3.4 아직 결정하지 않은 펌웨어 항목

- `SET HOLD` 기본값: 물체를 쥔 상태에서 coast로 풀리는지 벤치 시험 후 결정.
- 재파지(re-grip): 쥔 뒤 위치가 밀리면 다시 조이는 로직. 실제로 풀리는 것을 확인한 뒤에만 추가한다
  (불필요하면 PQ12 duty 20 % 규칙을 어기게 된다).

---

## 4. 젯슨 환경 준비

### 4.1 코드·설정 동기화

젯슨에 `configs/`가 없다. 최소한 다음이 필요하다.

```text
~/hjh_vision_hand/configs/hardware/hand_actuators.yaml
~/hjh_vision_hand/src/hand_control/**
~/hjh_vision_hand/scripts/hand_*.py
```

`scp` 또는 기존 번들 스크립트(`scripts/create_server_bundle.py`와 같은 방식)로 옮긴다.
젯슨의 `.venv`에 pyserial 3.5가 이미 있으므로 추가 설치는 없다. 다만 로컬 Windows에는
pyserial이 없어 `requirements-runtime.txt`(신규)에 명시하고 로컬에서만 설치한다.

### 4.2 ModemManager 회피 (중요)

젯슨의 ModemManager가 **active**다. `/dev/ttyACM0`이 나타나면 모뎀인지 확인하려고 포트를 열고
AT 명령을 밀어 넣는다. 결과로 (a) 우리 프로그램이 포트를 못 열거나, (b) 펌웨어 파서에 쓰레기
문자가 들어간다. 서비스를 끄지 않고 **이 장치만 무시**하게 udev 규칙을 추가한다.
Nano 33 IoT의 USB 식별자는 빌드 정의에서 확인했다: **VID 0x2341, PID 0x8057**.

`/etc/udev/rules.d/99-brunel-hand.rules`:

```text
# Arduino Nano 33 IoT (Brunel Hand 제어 보드)
# ModemManager가 AT 명령을 보내지 않게 하고, 고정 이름 /dev/brunel_hand 을 만든다.
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="8057", \
  ENV{ID_MM_DEVICE_IGNORE}="1", ENV{ID_MM_PORT_IGNORE}="1", \
  SYMLINK+="brunel_hand", MODE="0660", GROUP="dialout"
```

적용: `sudo udevadm control --reload-rules && sudo udevadm trigger`
(Jetson에서는 `sudo` 사용이 허용된다. 학습 서버에서는 금지.)

고정 심볼릭 링크가 생기면 포트 번호가 바뀌어도 `--port /dev/brunel_hand`로 고정할 수 있다.
카메라·RealSense가 함께 붙는 환경이라 번호 고정은 실질적인 이득이 있다.

### 4.3 포트 취급 규칙

- pyserial은 `exclusive=True`로 열어 두 프로세스가 동시에 잡는 것을 막는다.
- **1200 baud로 열면 안 된다.** SAMD 보드는 1200 bps 접속을 부트로더 진입 신호로 해석한다.
- USB 허브 전원 부족이나 진동으로 재열림이 필요할 수 있으므로, 클라이언트는 재연결 시
  `ID` → `STOP` → `SET ...` 순서를 다시 수행한다(연결 상태를 가정하지 않는다).
- 젯슨과 손 컨트롤러의 **10 V 모터 전원은 별개**다. 젯슨에서 전원을 켜고 끌 수 없으므로
  전원 인가는 사용자가 직접 한다(`AGENTS.md` 8절).

---

## 5. 젯슨 쪽 파이썬 패키지 설계

`AGENTS.md` 권장 구조의 `src/hand_control/`에 둔다. 보드 펌웨어는 `firmware/`에 유지.

| 파일 | 역할 | 시험 방법 |
|---|---|---|
| `protocol.py` | 명령 문자열 생성, 응답/이벤트 줄 파싱. **입출력 없음** (`OK`/`ERR`/`EVT` → dataclass) | 순수 단위 테스트 |
| `presets.py` | `hand_actuators.yaml` 로드·검증(`calibration_confirmed`, null 금지), 닫힘 비율 ↔ ADC 환산, 프리셋 조회 | 단위 테스트 |
| `link.py` | `SerialLink`(pyserial, 수신 스레드 + 큐, 타임아웃) / `MockLink`(펌웨어 동작 모사: 시간에 따라 위치 이동, `EVT DONE`/`STALL` 생성) | `MockLink`로 하드웨어 없이 |
| `client.py` | `HandClient`: `connect()`(ID 검증→STOP→SET LEGACY/HB/CAL), `stop()`, `open()/close()`, `move_fraction()`, `move_preset()`, `wait_done(timeout)`, `status()`, `with` 종료 시 항상 `STOP` | `MockLink` 통합 테스트 |
| `state_machine.py` | (마지막 단계) `PROJECT_CONTEXT.md` 12절 상태 머신 | 모사 `Decision` 시퀀스 |

CLI(젯슨에서 직접 쓰는 도구):

- `scripts/hand_cli.py --port /dev/brunel_hand` — 대화식/일회성 명령, 모든 송수신을
  `outputs/hand_tests/*.log`에 기록. `--mock`으로 하드웨어 없이 흐름 점검.
- `scripts/run_hand_presets.py --preset WRAP --repeat 5` — 프리셋 반복 시험과 결과 JSONL 기록.

설계 규칙:

- 카메라 루프를 막지 않도록 **수신은 스레드, 대기는 명시적 타임아웃**. 무한 대기 금지.
- 예외·`KeyboardInterrupt`·프로세스 종료(`atexit`/`finally`) 모두에서 `STOP` 전송.
- 보정값이 `null`이거나 `calibration_confirmed: false`면 **구동 자체를 거부**한다.
- 모든 시도를 로그로 남긴다(사용자 실측 없이 성능·성공률을 만들지 않는다는 규칙과 직결).

---

## 6. 실행 순서

각 단계는 이전 단계의 **확인 결과**가 있어야 넘어간다.

### A. 폐루프 펌웨어 벤치 시험 (Windows, 사용자 필요) — 젯슨 작업의 선행 조건

1. 터미널 종료 후 `firmware/hand_closed_loop` 업로드.
2. `ID` → `OK ID hand_closed_loop fw 2 ...` 확인(응답 없으면 업로드 실패).
3. `POS`, `MOVE 0 3000`, `FRAC 0 500`, `OPEN 0`, `FRAC 3 300`(엄지), `CLOSE`/`OPEN`.
4. 스톨 확인(손으로 막고 `CLOSE 0` → `EVT STALL`), `STOP` 즉시성 확인.
5. 쥔 상태에서 coast로 풀리는지 확인 → `SET HOLD` 기본값 결정.
   완료 기준: 목표 ± deadband 도달, 스톨·타임아웃 정상 동작, `STOP` 즉시 반응.

### B. 파이썬 패키지 구현 (하드웨어 불필요)

`protocol.py` → `presets.py` → `link.py`(Mock 포함) → `client.py`, `tests/` 3종 추가.
완료 기준: `python -m unittest discover -s tests -p "test_hand_*.py"` 통과,
`scripts/hand_cli.py --mock`으로 연결·프리셋·정지 흐름 재현.

### C. Windows 실물 연결 시험

`scripts/hand_cli.py --port COM10`으로 A단계와 같은 동작을 파이썬에서 재현.
heartbeat 시험: 이동 중 프로세스를 강제 종료 → `EVT HEARTBEAT_STOP` 확인.
완료 기준: 파이썬에서 낸 명령 결과가 A단계 수동 결과와 일치.

### D. 젯슨 이관

udev 규칙 적용 → 손 USB를 젯슨에 연결 → `/dev/brunel_hand` 확인 →
`configs/`·`src/hand_control/`·`scripts/` 동기화 → C단계와 같은 시험을 젯슨에서 반복.
완료 기준: 젯슨에서 `OPEN`/`CLOSE`/프리셋이 Windows와 동일하게 동작, 로그 저장.

### E. 프리셋 확정

`PRECISION`/`WRAP`/`POWER`와 각 `PRE_SHAPE`의 손가락별 닫힘 비율을 시행착오로 정해
YAML에 기록. 물체(머그 손잡이·몸통, 펜)로 반복 시험하고 성공률을 로그로 남긴다.
완료 기준: `PROJECT_CONTEXT.md` 1주차 기준 — "손의 세 자세가 명령별로 반복 구동된다".

### F. 비전 결합

`state_machine.py` 추가 → `run_camera_loop(..., hand_controller=None)` 훅 →
`scripts/run_realtime_seg.py --hand-port ... | --hand-mock`.
`CLOSE`는 기본적으로 사용자 키 트리거, 자동 모드는 설정으로 분리.
`ALIGN`/`NO_TARGET`/손 추적 소실/마스크 소실 시 정지.

---

## 7. 안전 규칙 (젯슨 운용 시 추가되는 것)

- 젯슨은 모터 전원을 제어하지 못한다. **전원 인가·차단은 사람이 한다.**
- 원격(SSH)에서 구동 명령을 내리지 않는다. 손이 움직이는 시험은 사람이 손 옆에 있을 때만.
  SSH로는 로그 확인·코드 배포·`STOP`까지만 한다.
- 자동 실행(부팅 시 서비스 등록, cron)은 만들지 않는다. 사람이 명시적으로 실행한다.
- heartbeat는 항상 켠다(`SET HB`). 켜지 않으면 호스트 사망 시 최대 3.5 s 구동이 이어진다.
- 연결 직후 항상 `ID`로 펌웨어를 검증하고, 불일치면 구동 명령을 보내지 않는다.

---

## 8. 사용자 확인·결정이 필요한 항목

1. **A단계 벤치 시험 결과** (로그) — 이것 없이는 B단계 이후로 못 넘어간다.
2. 쥔 상태에서 손가락이 풀리는지 → `SET HOLD` 기본값.
3. 젯슨에 udev 규칙을 추가해도 되는지(sudo 사용, ModemManager 회피).
4. 손 USB를 젯슨에 연결할 시점(현재 Windows 사용 중). 두 곳을 번갈아 쓸지, 젯슨 고정할지.
5. 젯슨 동기화 방법: `scp` 직접 / 번들 스크립트 / 별도 배포 스크립트 신설.
6. `requirements-runtime.txt` 신설(pyserial) — 다른 선호가 없으면 이대로 진행.
