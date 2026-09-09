// Brunel Hand 폐루프 펌웨어 보정값·제어 매개변수 (컴파일 기본값)
//
// 단일 진실은 configs/hardware/hand_actuators.yaml 이다. 이 파일의 값은 그 YAML에서
// 옮겨 적은 것이며, 호스트는 연결 시 SET CAL 로 덮어쓸 수 있다.
// 값 출처: outputs/hand_tests/serial_20260821_215011.log 등 (2026-08-21 실측)
//
// 인덱스: 0 검지, 1 중지, 2 약지·소지, 3 엄지

#pragma once

// ---- 보정값 (손가락별) ----
// 소프트 한계: 제어에서 허용하는 목표 범위. 기계 한계에서 80~100 안쪽.
const int CAL_SOFT_LOW[4]  = {450, 520, 520, 420};
const int CAL_SOFT_HIGH[4] = {3330, 3400, 3420, 3450};
// 기계 한계(관측값). 센서 이상 판정(범위 이탈)에만 쓴다.
const int CAL_HARD_LOW[4]  = {333, 432, 422, 309};
const int CAL_HARD_HIGH[4] = {3457, 3500, 3550, 3552};
// 펄스 스케치의 "방향 N"을 주면 ADC가 증가하는 N (1 또는 2). 중지만 2.
const int CAL_INCREASE_DIR[4] = {1, 2, 1, 1};
// 펼친 상태가 높은 ADC인지. 엄지만 반대(낮은 ADC = 펼침).
const bool CAL_EXTENDED_IS_HIGH[4] = {true, true, true, false};
// 1회 이동 상한(ms). 전 행정 실측의 약 2배. 엄지는 느리다.
const unsigned long CAL_MAX_MOVE_MS[4] = {3000, 3000, 3000, 3500};
// true면 부팅 직후부터 위 값으로 보정된 상태로 시작한다. false면 SET CAL 전까지 UNCAL.
const bool CAL_PRESENT = true;

// ---- 제어 매개변수 ----
// 목표 허용 오차. 2026-08-24 실측: 50이면 평균 오차 45 count, 25면 20, 12면 7이고
// 세 값 모두 진동 없이 DONE으로 끝났다(도달 시간은 305→349 ms). 20을 기본으로 둔다.
const int DEADBAND_ADC = 20;
const unsigned long LOOP_PERIOD_MS = 5;      // 이동 중 제어 주기
const int MOVE_SAMPLES = 4;                  // 이동 중 ADC 평균 샘플 수 (속도 우선)
const int REPORT_SAMPLES = 8;                // 정지 보고용 ADC 평균 샘플 수
const unsigned long STALL_WINDOW_MS = 200;   // 이 시간 동안
const int STALL_MIN_DELTA_ADC = 60;          // 목표 방향으로 이만큼도 못 가면 스톨
const unsigned long REST_AFTER_MOVE_MS = 500;// 이동 종료 후 같은 손가락 재구동 대기 (duty 보호)
const int SENSOR_FAULT_MARGIN_ADC = 200;     // 기계 한계 밖으로 이만큼 벗어나면 센서 이상
const int MAX_REVERSALS = 2;                 // 오버슛 보정 방향 전환 허용 횟수
const unsigned long MAX_PULSE_MS = 300;      // PULSE 명령 최대 길이
const unsigned long HEARTBEAT_DEFAULT_MS = 0;// 0 = 꺼짐. 호스트가 SET HB 로 켠다
// 도달/스톨 후 brake(H/H)로 유지할지. 2026-08-24 실측: coast로 두면 검지가 3초에
// 279 count 처졌고 brake는 40 count였다. 파지 유지에 필수이므로 기본으로 켠다.
const bool HOLD_BRAKE_DEFAULT = true;
// 단일 문자 명령(1~8, s, p, h) 허용 여부. 사람이 시리얼 모니터로 시험할 때 편하므로 기본 허용.
// 젯슨 등 헤드리스 호스트는 연결 시 SET LEGACY 0 으로 끈다 (잡음 한 글자로 모터가 도는 것을 막는다).
const bool LEGACY_ENABLED_DEFAULT = true;
