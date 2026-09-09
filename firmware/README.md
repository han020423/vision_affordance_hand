# firmware

Arduino 계열 보드에 올리는 펌웨어 스케치 모음. Jetson 쪽 Python 시리얼 클라이언트는
`src/hand_control/`에 두고, 여기에는 보드에서 실행되는 코드만 둔다.

Arduino IDE 규칙상 각 스케치는 `폴더명/폴더명.ino` 쌍으로 유지한다.

| 폴더 | 보드 | 용도 |
|---|---|---|
| `hand_control/` | Arduino Nano 33 IoT (SAMD21) | Brunel Hand V2.0용 PQ12-100-12-P ×4 + DRV8833 ×2 제어. 현재는 100 ms 펄스 수동 테스트 단계 |
| `hand_feedback_check/` | Arduino Nano 33 IoT (SAMD21) | PQ12 위치 피드백(A0~A3) 진단 전용. 모터 IN 핀은 LOW로 고정하고 ADC 값·평균·최소/최대만 주기 출력 |
| `hand_closed_loop/` | Arduino Nano 33 IoT (SAMD21) | **현재 주력 (v2).** 위치 피드백 기반 폐루프 제어. 줄 단위 텍스트 프로토콜(ID/MOVE/FRAC/OPEN/CLOSE/STOP/PULSE/SET/GET), 스톨·타임아웃·센서 이상·heartbeat 안전장치, 구 스케치 `1`~`8`/`s`/`p` 호환(`SET LEGACY 0`으로 차단 가능). 보정값은 `hand_config.h`(출처: `configs/hardware/hand_actuators.yaml`) |

## 컴파일 검증 (업로드 없이)

Arduino IDE에 포함된 arduino-cli로 업로드 전에 컴파일만 확인할 수 있다.

```bash
"$LOCALAPPDATA/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe" compile --fqbn arduino:samd:nano_33_iot --warnings all firmware/hand_closed_loop
```

## 업로드 시 주의

- `scripts/hand_serial_terminal.py`나 IDE 시리얼 모니터가 포트를 잡고 있으면 업로드가 "No device found"로
  실패한다. 터미널을 먼저 끄고 업로드한 뒤 다시 켠다.

## hand_control 빌드·업로드

- 보드: Arduino SAMD Boards → Arduino NANO 33 IoT
- 시리얼: 115200 baud, 줄 끝 문자 없음 또는 무시됨(단일 문자 명령)
- 모터 전원(10 V)은 반드시 사용자가 배선·전원 상태를 확인한 뒤 켠다. 업로드·시리얼
  모니터 연결은 10 V를 끈 상태에서 해도 된다.

## 업로드 확인 방법

업로드가 실제로 됐는지는 시리얼에서 `ID`를 보내 확인한다.

- `OK ID hand_closed_loop fw 2 proto 1 ...` → 폐루프 펌웨어가 올라가 있다.
- 아무 응답이 없다 → 옛 펄스 스케치(`hand_control/`)가 그대로 있다. 이 상태에서 `moveall ...`
  같은 문장을 보내면 숫자가 낱개 펄스 명령으로 해석되어 모터가 제멋대로 움직인다.

## 젯슨 연결

젯슨에서 이 보드를 제어하는 절차와 udev 규칙은 `docs/jetson_hand_control_plan.md`에 있다.
