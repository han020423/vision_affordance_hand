// Brunel Hand V2.0 폐루프 제어 펌웨어 (Arduino Nano 33 IoT + DRV8833 ×2 + PQ12-100-12-P ×4)
//
// - 손가락별 위치 피드백(A0~A3)으로 목표 ADC까지 연속 구동하고 deadband 안에 들어오면 정지한다.
// - 모든 이동에 최대 구동시간, 스톨(진행 없음) 감지, 센서 범위 이탈 감지가 붙는다.
// - STOP 은 어떤 상태에서도 즉시 전체 coast.
// - 이동 중 heartbeat(호스트 명령) 가 끊기면 전체 정지 (SET HB 로 켬, 기본 꺼짐).
// - PQ12-P 에는 종단 리미트 스위치가 없으므로 목표는 소프트 한계 안으로만 허용한다.
//
// 배선/핀: MOTOR_IN1={2,4,6,8}, MOTOR_IN2={3,5,7,9}, FEEDBACK={A0,A1,A2,A3}
// 인덱스: 0 검지, 1 중지, 2 약지·소지, 3 엄지
// 보정값·매개변수: hand_config.h (단일 진실은 configs/hardware/hand_actuators.yaml)
//
// 시리얼 115200, 줄 단위 ASCII. 응답 "OK ..." / "ERR <code> ...", 비동기 "EVT ...".
// 명령:
//   PING                         -> OK PONG <ms>
//   STOP  (또는 s)               -> 전체 coast, 이동 취소
//   POS   (또는 p)               -> OK POS a0 a1 a2 a3          (8회 평균)
//   STATUS                       -> OK STATUS i:STATE:adc:target ...
//   MOVE <i> <adc>               -> 손가락 i 를 목표 ADC 로 (소프트 한계 안)
//   MOVEALL <a0> <a1> <a2> <a3>  -> 네 손가락 동시 목표 (-1 = 그대로)
//   FRAC <i> <0..1000>           -> 닫힘 비율(천분율)로 이동. 0 = 펼침, 1000 = 접힘(소프트 한계)
//   FRACALL <f0> <f1> <f2> <f3>  -> 네 손가락 동시 비율 (-1 = 그대로)
//   OPEN [i]  / CLOSE [i]        -> 전체 또는 한 손가락 펼침/접힘
//   PULSE <i> <dir 1|2> <ms>     -> 시간 기반 펄스 (<= MAX_PULSE_MS, 소프트 한계 밖으로는 거부)
//   1~8 (구 스케치 호환)         -> PULSE (i=(n-1)/2, dir=(n-1)%2+1, 100 ms)
//   SET CAL <i> <low> <high> <incdir 1|2> <exthigh 0|1> [maxmove_ms]
//   SET HB <ms>                  -> heartbeat 끊김 판정 시간 (0 = 끔)
//   SET DEADBAND <adc>
//   SET HOLD <0|1>               -> 도달/스톨 후 brake 유지 여부
//   SET LEGACY <0|1>             -> 단일 문자 명령 허용 여부 (헤드리스 운용 시 0 권장)
//   ID                           -> 펌웨어·프로토콜 식별 (호스트가 연결 시 확인)
//   GET CAL / GET PARAM
//   HELP (또는 h)
// 이벤트:
//   EVT DONE|STALL|TIMEOUT|FAULT <i> <adc>,  EVT PULSE_DONE <i> <adc>,  EVT HEARTBEAT_STOP,  EVT BOOT

#include "hand_types.h"
#include "hand_config.h"

// 젯슨 등 헤드리스 호스트 운용 참고
// - Nano 33 IoT의 Serial은 네이티브 USB CDC다. 호스트가 포트를 열어도 보드는 리셋되지 않으므로
//   연결 직후 `EVT BOOT`를 기다리지 말고 `ID`로 펌웨어를 확인해야 한다.
// - `if (Serial)`(CDC 연결 확인)은 SAMD 코어 구현에 delay(10)이 들어 있어 제어 루프에서 쓰면 안 된다.
//   호스트 소실은 heartbeat(`SET HB`)로 감지한다.
// - 호스트가 읽지 않는 상태에서 출력하면 USB 전송이 최대 70 ms(TX_TIMEOUT_MS) 지연될 수 있다.
//   이벤트 출력은 모터를 먼저 정지시킨 뒤 수행하므로 지연 중 구동이 이어지지는 않는다.
// - 잡음이나 오타 한 글자가 모터를 움직이지 않도록, 호스트는 연결 시 `SET LEGACY 0`으로
//   단일 문자 명령을 끄는 것을 권장한다.

const char* FIRMWARE_NAME = "hand_closed_loop";
const int FIRMWARE_VERSION = 2;
const int PROTOCOL_VERSION = 1;

const int FINGER_COUNT = 4;
const int MOTOR_IN1[FINGER_COUNT] = {2, 4, 6, 8};
const int MOTOR_IN2[FINGER_COUNT] = {3, 5, 7, 9};
const int FEEDBACK_PIN[FINGER_COUNT] = {A0, A1, A2, A3};
const char* FINGER_NAME[FINGER_COUNT] = {"index", "middle", "ring_little", "thumb"};

Finger fingers[FINGER_COUNT];

// 런타임 매개변수 (SET 으로 변경 가능)
int deadbandAdc = DEADBAND_ADC;
unsigned long heartbeatMs = HEARTBEAT_DEFAULT_MS;
bool holdBrake = HOLD_BRAKE_DEFAULT;
bool legacyEnabled = LEGACY_ENABLED_DEFAULT;

unsigned long lastRxMs = 0;
unsigned long lastLoopMs = 0;

const int LINE_BUFFER_SIZE = 96;
char lineBuffer[LINE_BUFFER_SIZE];
int lineLength = 0;

// ---------------------------------------------------------------- 저수준 구동

void coastFinger(int i) {
  digitalWrite(MOTOR_IN1[i], LOW);
  digitalWrite(MOTOR_IN2[i], LOW);
}

void brakeFinger(int i) {
  digitalWrite(MOTOR_IN1[i], HIGH);
  digitalWrite(MOTOR_IN2[i], HIGH);
}

// 펄스 스케치와 같은 의미의 "방향 번호"로 구동한다. 1: IN1=H/IN2=L, 2: IN1=L/IN2=H
void driveDirectionNumber(int i, int directionNumber) {
  if (directionNumber == 1) {
    digitalWrite(MOTOR_IN2[i], LOW);
    digitalWrite(MOTOR_IN1[i], HIGH);
  } else {
    digitalWrite(MOTOR_IN1[i], LOW);
    digitalWrite(MOTOR_IN2[i], HIGH);
  }
}

// ADC 증감 방향(+1 증가, -1 감소)으로 구동한다.
void driveAdcDirection(int i, int adcDirection) {
  int increase = fingers[i].increaseDir;
  int decrease = (increase == 1) ? 2 : 1;
  driveDirectionNumber(i, adcDirection > 0 ? increase : decrease);
}

int readAdc(int i, int samples) {
  long sum = 0;
  for (int k = 0; k < samples; k++) {
    sum += analogRead(FEEDBACK_PIN[i]);
  }
  return (int)(sum / samples);
}

FingerState idleState(int i) {
  return fingers[i].calibrated ? ST_IDLE : ST_UNCAL;
}

const char* stateName(FingerState state) {
  switch (state) {
    case ST_UNCAL: return "UNCAL";
    case ST_IDLE: return "IDLE";
    case ST_WAIT: return "WAIT";
    case ST_MOVING: return "MOVING";
    case ST_PULSE: return "PULSE";
    case ST_DONE: return "DONE";
    case ST_STALL: return "STALL";
    case ST_TIMEOUT: return "TIMEOUT";
    case ST_FAULT: return "FAULT";
  }
  return "?";
}

bool isActive(int i) {
  FingerState s = fingers[i].state;
  return s == ST_MOVING || s == ST_WAIT || s == ST_PULSE;
}

bool anyActive() {
  for (int i = 0; i < FINGER_COUNT; i++) {
    if (isActive(i)) return true;
  }
  return false;
}

// 전체 즉시 정지 (coast). 진행 중인 이동·펄스·대기를 모두 취소한다.
void stopAll() {
  for (int i = 0; i < FINGER_COUNT; i++) {
    coastFinger(i);
    fingers[i].dir = 0;
    fingers[i].state = idleState(i);
  }
}

// ---------------------------------------------------------------- 출력 도우미

void printEvent(const char* name, int i, int adc) {
  Serial.print("EVT ");
  Serial.print(name);
  Serial.print(' ');
  Serial.print(i);
  Serial.print(' ');
  Serial.println(adc);
}

void printError(const char* code, const char* detail) {
  Serial.print("ERR ");
  Serial.print(code);
  if (detail != nullptr && detail[0] != '\0') {
    Serial.print(' ');
    Serial.print(detail);
  }
  Serial.println();
}

void printPositions() {
  Serial.print("OK POS");
  for (int i = 0; i < FINGER_COUNT; i++) {
    int adc = (fingers[i].state == ST_MOVING) ? fingers[i].lastAdc : readAdc(i, REPORT_SAMPLES);
    fingers[i].lastAdc = adc;
    Serial.print(' ');
    Serial.print(adc);
  }
  Serial.println();
}

void printStatus() {
  Serial.print("OK STATUS");
  for (int i = 0; i < FINGER_COUNT; i++) {
    Finger& f = fingers[i];
    int adc = (f.state == ST_MOVING) ? f.lastAdc : readAdc(i, REPORT_SAMPLES);
    f.lastAdc = adc;
    Serial.print(' ');
    Serial.print(i);
    Serial.print(':');
    Serial.print(stateName(f.state));
    Serial.print(':');
    Serial.print(adc);
    Serial.print(':');
    if (f.state == ST_MOVING || f.state == ST_WAIT) {
      Serial.print(f.target);
    } else {
      Serial.print('-');
    }
  }
  Serial.println();
}

void printCalibration() {
  Serial.print("OK CAL");
  for (int i = 0; i < FINGER_COUNT; i++) {
    Finger& f = fingers[i];
    Serial.print(' ');
    Serial.print(i);
    Serial.print(':');
    if (!f.calibrated) {
      Serial.print("UNCAL");
      continue;
    }
    Serial.print(f.softLow);
    Serial.print(':');
    Serial.print(f.softHigh);
    Serial.print(':');
    Serial.print(f.increaseDir);
    Serial.print(':');
    Serial.print(f.extendedIsHigh ? 1 : 0);
    Serial.print(':');
    Serial.print(f.maxMoveMs);
  }
  Serial.println();
}

// 호스트가 연결 직후 어떤 펌웨어인지 확인하는 용도.
// 옛 펄스 스케치는 ID 에 아무 응답도 하지 않으므로, 응답이 없으면 잘못된 펌웨어다.
void printIdentity() {
  Serial.print("OK ID ");
  Serial.print(FIRMWARE_NAME);
  Serial.print(" fw ");
  Serial.print(FIRMWARE_VERSION);
  Serial.print(" proto ");
  Serial.print(PROTOCOL_VERSION);
  Serial.print(" fingers ");
  Serial.print(FINGER_COUNT);
  Serial.print(" adcbits 12 uptime ");
  Serial.println(millis());
}

void printParams() {
  Serial.print("OK PARAM deadband=");
  Serial.print(deadbandAdc);
  Serial.print(" hb=");
  Serial.print(heartbeatMs);
  Serial.print(" hold=");
  Serial.print(holdBrake ? 1 : 0);
  Serial.print(" legacy=");
  Serial.print(legacyEnabled ? 1 : 0);
  Serial.print(" stall_window=");
  Serial.print(STALL_WINDOW_MS);
  Serial.print(" stall_delta=");
  Serial.print(STALL_MIN_DELTA_ADC);
  Serial.print(" rest=");
  Serial.print(REST_AFTER_MOVE_MS);
  Serial.print(" loop=");
  Serial.println(LOOP_PERIOD_MS);
}

void printHelp() {
  Serial.println("OK HELP");
  Serial.println("# ID | PING | STOP(s) | POS(p) | STATUS | HELP(h)");
  Serial.println("# MOVE <i> <adc> | MOVEALL a0 a1 a2 a3 (-1=유지)");
  Serial.println("# FRAC <i> <0..1000> | FRACALL f0 f1 f2 f3 (0=펼침, 1000=접힘)");
  Serial.println("# OPEN [i] | CLOSE [i]");
  Serial.println("# PULSE <i> <dir 1|2> <ms<=300> | 1~8 = 구 스케치 100ms 펄스");
  Serial.println("# SET CAL <i> <low> <high> <incdir> <exthigh> [maxmove] | SET HB <ms> | SET DEADBAND <adc> | SET HOLD <0|1> | SET LEGACY <0|1>");
  Serial.println("# GET CAL | GET PARAM");
  Serial.println("# 인덱스: 0 검지, 1 중지, 2 약지·소지, 3 엄지");
}

// ---------------------------------------------------------------- 이동 제어

// 이동 종료 처리: 모터 정지, 상태 기록, 이벤트 출력, 휴지 시간 설정
void finishMove(int i, FingerState state, const char* name, int adc) {
  Finger& f = fingers[i];
  bool hold = holdBrake && (state == ST_DONE || state == ST_STALL);
  if (hold) {
    brakeFinger(i);
  } else {
    coastFinger(i);
  }
  f.dir = 0;
  f.state = state;
  f.restUntilMs = millis() + REST_AFTER_MOVE_MS;
  printEvent(name, i, adc);
}

void beginMotion(int i, unsigned long now) {
  Finger& f = fingers[i];
  f.moveStartMs = now;
  f.lastProgressMs = now;
  f.lastAdc = readAdc(i, MOVE_SAMPLES);
  f.lastProgressAdc = f.lastAdc;
  f.dir = 0;
  f.reversals = 0;
  f.state = ST_MOVING;
}

// 목표 설정. 성공 시 true, 실패 시 ERR 를 출력하고 false.
bool startMove(int i, int target) {
  Finger& f = fingers[i];
  if (!f.calibrated) {
    printError("UNCAL", FINGER_NAME[i]);
    return false;
  }
  if (target < f.softLow || target > f.softHigh) {
    printError("RANGE", FINGER_NAME[i]);
    return false;
  }
  if (f.state == ST_PULSE) {
    printError("BUSY", "pulse");
    return false;
  }
  unsigned long now = millis();
  f.target = target;
  if (f.state == ST_MOVING) {
    // 이동 중 목표 변경: 시작 시각은 유지해 연속 구동시간 상한을 지킨다.
    f.dir = 0;
    f.reversals = 0;
    f.lastProgressMs = now;
    f.lastProgressAdc = f.lastAdc;
    return true;
  }
  if ((long)(f.restUntilMs - now) > 0) {
    f.state = ST_WAIT;
    return true;
  }
  beginMotion(i, now);
  return true;
}

// 닫힘 비율(천분율) -> 목표 ADC
int fractionToTarget(int i, int permille) {
  Finger& f = fingers[i];
  int openAdc = f.extendedIsHigh ? f.softHigh : f.softLow;
  int closedAdc = f.extendedIsHigh ? f.softLow : f.softHigh;
  long span = (long)closedAdc - (long)openAdc;
  return (int)(openAdc + span * permille / 1000);
}

bool startPulse(int i, int directionNumber, unsigned long ms) {
  Finger& f = fingers[i];
  if (directionNumber != 1 && directionNumber != 2) {
    printError("ARG", "dir must be 1 or 2");
    return false;
  }
  if (ms == 0 || ms > MAX_PULSE_MS) {
    printError("ARG", "ms out of range");
    return false;
  }
  if (f.state == ST_MOVING || f.state == ST_WAIT || f.state == ST_PULSE) {
    printError("BUSY", FINGER_NAME[i]);
    return false;
  }
  int adc = readAdc(i, MOVE_SAMPLES);
  f.lastAdc = adc;
  if (f.calibrated) {
    int adcDirection = (directionNumber == f.increaseDir) ? +1 : -1;
    if ((adcDirection > 0 && adc >= f.softHigh) || (adcDirection < 0 && adc <= f.softLow)) {
      printError("LIMIT", FINGER_NAME[i]);
      return false;
    }
  }
  driveDirectionNumber(i, directionNumber);
  f.state = ST_PULSE;
  f.pulseEndMs = millis() + ms;
  return true;
}

void updateFinger(int i, unsigned long now) {
  Finger& f = fingers[i];

  if (f.state == ST_WAIT) {
    if ((long)(now - f.restUntilMs) >= 0) {
      beginMotion(i, now);
    }
    return;
  }

  if (f.state == ST_PULSE) {
    if ((long)(now - f.pulseEndMs) >= 0) {
      coastFinger(i);
      f.state = idleState(i);
      f.lastAdc = readAdc(i, MOVE_SAMPLES);
      printEvent("PULSE_DONE", i, f.lastAdc);
    }
    return;
  }

  if (f.state != ST_MOVING) return;

  int adc = readAdc(i, MOVE_SAMPLES);
  f.lastAdc = adc;

  if (adc < f.hardLow - SENSOR_FAULT_MARGIN_ADC || adc > f.hardHigh + SENSOR_FAULT_MARGIN_ADC) {
    finishMove(i, ST_FAULT, "FAULT", adc);
    return;
  }

  int error = f.target - adc;
  if (abs(error) <= deadbandAdc) {
    finishMove(i, ST_DONE, "DONE", adc);
    return;
  }

  int wanted = (error > 0) ? +1 : -1;
  if (f.dir == 0) {
    f.dir = wanted;
    driveAdcDirection(i, wanted);
    f.lastProgressAdc = adc;
    f.lastProgressMs = now;
  } else if (wanted != f.dir) {
    // 오버슛: 제한 횟수 안에서만 되돌린다. 초과하면 그 자리에서 종료한다.
    f.reversals++;
    if (f.reversals > MAX_REVERSALS) {
      finishMove(i, ST_DONE, "DONE", adc);
      return;
    }
    f.dir = wanted;
    driveAdcDirection(i, wanted);
    f.lastProgressAdc = adc;
    f.lastProgressMs = now;
  } else {
    if ((long)(adc - f.lastProgressAdc) * f.dir >= STALL_MIN_DELTA_ADC) {
      f.lastProgressAdc = adc;
      f.lastProgressMs = now;
    } else if (now - f.lastProgressMs >= STALL_WINDOW_MS) {
      finishMove(i, ST_STALL, "STALL", adc);
      return;
    }
  }

  if (now - f.moveStartMs >= f.maxMoveMs) {
    finishMove(i, ST_TIMEOUT, "TIMEOUT", adc);
    return;
  }
}

// ---------------------------------------------------------------- 명령 처리

bool parseInt(const char* token, long* out) {
  if (token == nullptr || *token == '\0') return false;
  char* end = nullptr;
  long value = strtol(token, &end, 10);
  if (end == token || *end != '\0') return false;
  *out = value;
  return true;
}

bool parseFingerIndex(const char* token, int* out) {
  long value;
  if (!parseInt(token, &value) || value < 0 || value >= FINGER_COUNT) {
    printError("ARG", "finger index 0..3");
    return false;
  }
  *out = (int)value;
  return true;
}

void toUpper(char* s) {
  for (; *s; s++) {
    if (*s >= 'a' && *s <= 'z') *s = *s - 'a' + 'A';
  }
}

// 구 스케치 단일 문자 명령
void handleLegacyChar(char c) {
  if (c >= '1' && c <= '8') {
    int n = c - '1';
    int i = n / 2;
    int directionNumber = (n % 2) + 1;
    if (startPulse(i, directionNumber, 100)) {
      Serial.print("OK PULSE ");
      Serial.print(i);
      Serial.print(' ');
      Serial.print(directionNumber);
      Serial.println(" 100");
    }
  } else if (c == 's' || c == 'S') {
    stopAll();
    Serial.println("OK STOP");
  } else if (c == 'p' || c == 'P') {
    printPositions();
  } else if (c == 'h' || c == 'H') {
    printHelp();
  }
}

bool isLegacyLine(const char* line) {
  for (const char* p = line; *p; p++) {
    if (!((*p >= '1' && *p <= '8') || *p == 's' || *p == 'S' || *p == 'p' || *p == 'P' || *p == 'h' || *p == 'H')) {
      return false;
    }
  }
  return true;
}

void handleMoveAll(bool fraction) {
  long values[FINGER_COUNT];
  for (int i = 0; i < FINGER_COUNT; i++) {
    char* token = strtok(nullptr, " \t");
    if (!parseInt(token, &values[i])) {
      printError("ARG", "need 4 values");
      return;
    }
  }
  // 먼저 전부 검증한 뒤 시작해서 일부만 움직이는 상황을 피한다.
  int targets[FINGER_COUNT];
  for (int i = 0; i < FINGER_COUNT; i++) {
    if (values[i] < 0) {
      targets[i] = -1;
      continue;
    }
    if (!fingers[i].calibrated) {
      printError("UNCAL", FINGER_NAME[i]);
      return;
    }
    if (fraction) {
      if (values[i] > 1000) {
        printError("ARG", "fraction 0..1000");
        return;
      }
      targets[i] = fractionToTarget(i, (int)values[i]);
    } else {
      targets[i] = (int)values[i];
      if (targets[i] < fingers[i].softLow || targets[i] > fingers[i].softHigh) {
        printError("RANGE", FINGER_NAME[i]);
        return;
      }
    }
    if (fingers[i].state == ST_PULSE) {
      printError("BUSY", "pulse");
      return;
    }
  }
  for (int i = 0; i < FINGER_COUNT; i++) {
    if (targets[i] >= 0) startMove(i, targets[i]);
  }
  Serial.print(fraction ? "OK FRACALL" : "OK MOVEALL");
  for (int i = 0; i < FINGER_COUNT; i++) {
    Serial.print(' ');
    Serial.print(targets[i]);
  }
  Serial.println();
}

void handleOpenClose(bool close) {
  char* token = strtok(nullptr, " \t");
  int permille = close ? 1000 : 0;
  if (token != nullptr) {
    int i;
    if (!parseFingerIndex(token, &i)) return;
    if (!fingers[i].calibrated) {
      printError("UNCAL", FINGER_NAME[i]);
      return;
    }
    int target = fractionToTarget(i, permille);
    if (startMove(i, target)) {
      Serial.print(close ? "OK CLOSE " : "OK OPEN ");
      Serial.print(i);
      Serial.print(' ');
      Serial.println(target);
    }
    return;
  }
  for (int i = 0; i < FINGER_COUNT; i++) {
    if (!fingers[i].calibrated) {
      printError("UNCAL", FINGER_NAME[i]);
      return;
    }
    if (fingers[i].state == ST_PULSE) {
      printError("BUSY", "pulse");
      return;
    }
  }
  Serial.print(close ? "OK CLOSE" : "OK OPEN");
  for (int i = 0; i < FINGER_COUNT; i++) {
    int target = fractionToTarget(i, permille);
    startMove(i, target);
    Serial.print(' ');
    Serial.print(target);
  }
  Serial.println();
}

void handleSet() {
  char* what = strtok(nullptr, " \t");
  if (what == nullptr) {
    printError("ARG", "SET CAL|HB|DEADBAND|HOLD|LEGACY");
    return;
  }
  toUpper(what);
  if (strcmp(what, "CAL") == 0) {
    int i;
    if (!parseFingerIndex(strtok(nullptr, " \t"), &i)) return;
    long low, high, incdir, exthigh, maxmove = -1;
    if (!parseInt(strtok(nullptr, " \t"), &low) || !parseInt(strtok(nullptr, " \t"), &high) ||
        !parseInt(strtok(nullptr, " \t"), &incdir) || !parseInt(strtok(nullptr, " \t"), &exthigh)) {
      printError("ARG", "SET CAL <i> <low> <high> <incdir> <exthigh> [maxmove]");
      return;
    }
    char* maxToken = strtok(nullptr, " \t");
    if (maxToken != nullptr && !parseInt(maxToken, &maxmove)) {
      printError("ARG", "maxmove");
      return;
    }
    if (low < 0 || high > 4095 || low >= high || (incdir != 1 && incdir != 2) || (exthigh != 0 && exthigh != 1)) {
      printError("ARG", "cal values");
      return;
    }
    if (isActive(i)) {
      printError("BUSY", FINGER_NAME[i]);
      return;
    }
    Finger& f = fingers[i];
    f.softLow = (int)low;
    f.softHigh = (int)high;
    f.increaseDir = (int)incdir;
    f.extendedIsHigh = (exthigh == 1);
    if (maxmove > 0) f.maxMoveMs = (unsigned long)maxmove;
    f.calibrated = true;
    f.state = ST_IDLE;
    Serial.print("OK CAL ");
    Serial.println(i);
    return;
  }
  long value;
  if (!parseInt(strtok(nullptr, " \t"), &value)) {
    printError("ARG", "value");
    return;
  }
  if (strcmp(what, "HB") == 0) {
    if (value < 0) { printError("ARG", "hb"); return; }
    heartbeatMs = (unsigned long)value;
    Serial.print("OK HB ");
    Serial.println(heartbeatMs);
  } else if (strcmp(what, "DEADBAND") == 0) {
    if (value < 5 || value > 500) { printError("ARG", "deadband 5..500"); return; }
    deadbandAdc = (int)value;
    Serial.print("OK DEADBAND ");
    Serial.println(deadbandAdc);
  } else if (strcmp(what, "HOLD") == 0) {
    holdBrake = (value != 0);
    Serial.print("OK HOLD ");
    Serial.println(holdBrake ? 1 : 0);
  } else if (strcmp(what, "LEGACY") == 0) {
    legacyEnabled = (value != 0);
    Serial.print("OK LEGACY ");
    Serial.println(legacyEnabled ? 1 : 0);
  } else {
    printError("ARG", "SET CAL|HB|DEADBAND|HOLD|LEGACY");
  }
}

void handleLine(char* line) {
  // 앞뒤 공백 제거
  while (*line == ' ' || *line == '\t') line++;
  int n = strlen(line);
  while (n > 0 && (line[n - 1] == ' ' || line[n - 1] == '\t')) line[--n] = '\0';
  if (n == 0) return;

  if (isLegacyLine(line)) {
    // STOP 은 어떤 설정에서도 통해야 하므로 s/S 는 legacy 를 꺼도 받는다.
    if (!legacyEnabled) {
      bool onlyStop = true;
      for (char* p = line; *p; p++) {
        if (*p != 's' && *p != 'S') { onlyStop = false; break; }
      }
      if (!onlyStop) {
        printError("LEGACY_OFF", line);
        return;
      }
    }
    for (char* p = line; *p; p++) handleLegacyChar(*p);
    return;
  }

  char* command = strtok(line, " \t");
  if (command == nullptr) return;
  toUpper(command);

  if (strcmp(command, "STOP") == 0) {
    stopAll();
    Serial.println("OK STOP");
  } else if (strcmp(command, "ID") == 0) {
    printIdentity();
  } else if (strcmp(command, "PING") == 0) {
    Serial.print("OK PONG ");
    Serial.println(millis());
  } else if (strcmp(command, "POS") == 0) {
    printPositions();
  } else if (strcmp(command, "STATUS") == 0) {
    printStatus();
  } else if (strcmp(command, "HELP") == 0) {
    printHelp();
  } else if (strcmp(command, "MOVE") == 0) {
    int i;
    long target;
    if (!parseFingerIndex(strtok(nullptr, " \t"), &i)) return;
    if (!parseInt(strtok(nullptr, " \t"), &target)) { printError("ARG", "MOVE <i> <adc>"); return; }
    if (startMove(i, (int)target)) {
      Serial.print("OK MOVE ");
      Serial.print(i);
      Serial.print(' ');
      Serial.println(target);
    }
  } else if (strcmp(command, "MOVEALL") == 0) {
    handleMoveAll(false);
  } else if (strcmp(command, "FRAC") == 0) {
    int i;
    long permille;
    if (!parseFingerIndex(strtok(nullptr, " \t"), &i)) return;
    if (!parseInt(strtok(nullptr, " \t"), &permille) || permille < 0 || permille > 1000) {
      printError("ARG", "FRAC <i> <0..1000>");
      return;
    }
    if (!fingers[i].calibrated) { printError("UNCAL", FINGER_NAME[i]); return; }
    int target = fractionToTarget(i, (int)permille);
    if (startMove(i, target)) {
      Serial.print("OK FRAC ");
      Serial.print(i);
      Serial.print(' ');
      Serial.println(target);
    }
  } else if (strcmp(command, "FRACALL") == 0) {
    handleMoveAll(true);
  } else if (strcmp(command, "OPEN") == 0) {
    handleOpenClose(false);
  } else if (strcmp(command, "CLOSE") == 0) {
    handleOpenClose(true);
  } else if (strcmp(command, "PULSE") == 0) {
    int i;
    long directionNumber, ms;
    if (!parseFingerIndex(strtok(nullptr, " \t"), &i)) return;
    if (!parseInt(strtok(nullptr, " \t"), &directionNumber) || !parseInt(strtok(nullptr, " \t"), &ms)) {
      printError("ARG", "PULSE <i> <dir> <ms>");
      return;
    }
    if (ms < 0) { printError("ARG", "ms"); return; }
    if (startPulse(i, (int)directionNumber, (unsigned long)ms)) {
      Serial.print("OK PULSE ");
      Serial.print(i);
      Serial.print(' ');
      Serial.print(directionNumber);
      Serial.print(' ');
      Serial.println(ms);
    }
  } else if (strcmp(command, "SET") == 0) {
    handleSet();
  } else if (strcmp(command, "GET") == 0) {
    char* what = strtok(nullptr, " \t");
    if (what != nullptr) toUpper(what);
    if (what != nullptr && strcmp(what, "CAL") == 0) {
      printCalibration();
    } else if (what != nullptr && strcmp(what, "PARAM") == 0) {
      printParams();
    } else {
      printError("ARG", "GET CAL|PARAM");
    }
  } else {
    printError("UNKNOWN", command);
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    lastRxMs = millis();
    if (c == '\n' || c == '\r') {
      lineBuffer[lineLength] = '\0';
      if (lineLength > 0) handleLine(lineBuffer);
      lineLength = 0;
    } else if (lineLength < LINE_BUFFER_SIZE - 1) {
      lineBuffer[lineLength++] = c;
    } else {
      // 너무 긴 줄은 버린다
      lineLength = 0;
      printError("ARG", "line too long");
    }
  }
}

// ---------------------------------------------------------------- setup / loop

void setup() {
  // 가장 먼저 모든 모터 입력을 LOW(coast)로 고정한다.
  for (int i = 0; i < FINGER_COUNT; i++) {
    pinMode(MOTOR_IN1[i], OUTPUT);
    pinMode(MOTOR_IN2[i], OUTPUT);
    digitalWrite(MOTOR_IN1[i], LOW);
    digitalWrite(MOTOR_IN2[i], LOW);
    pinMode(FEEDBACK_PIN[i], INPUT);
  }
  analogReadResolution(12);

  for (int i = 0; i < FINGER_COUNT; i++) {
    Finger& f = fingers[i];
    f.calibrated = CAL_PRESENT;
    f.softLow = CAL_SOFT_LOW[i];
    f.softHigh = CAL_SOFT_HIGH[i];
    f.hardLow = CAL_HARD_LOW[i];
    f.hardHigh = CAL_HARD_HIGH[i];
    f.increaseDir = CAL_INCREASE_DIR[i];
    f.extendedIsHigh = CAL_EXTENDED_IS_HIGH[i];
    f.maxMoveMs = CAL_MAX_MOVE_MS[i];
    f.state = f.calibrated ? ST_IDLE : ST_UNCAL;
    f.target = 0;
    f.dir = 0;
    f.reversals = 0;
    f.moveStartMs = 0;
    f.lastProgressMs = 0;
    f.lastProgressAdc = 0;
    f.restUntilMs = 0;
    f.pulseEndMs = 0;
    f.lastAdc = 0;
  }

  Serial.begin(115200);
  delay(500);
  Serial.print("EVT BOOT ");
  Serial.print(FIRMWARE_NAME);
  Serial.print(" fw ");
  Serial.println(FIRMWARE_VERSION);
  lastRxMs = millis();
}

void loop() {
  pollSerial();

  unsigned long now = millis();

  // heartbeat: 이동 중 호스트 명령이 끊기면 전체 정지
  if (heartbeatMs > 0 && anyActive() && (now - lastRxMs) > heartbeatMs) {
    stopAll();
    Serial.println("EVT HEARTBEAT_STOP");
    lastRxMs = now;  // 반복 출력 방지
  }

  if (now - lastLoopMs >= LOOP_PERIOD_MS) {
    lastLoopMs = now;
    for (int i = 0; i < FINGER_COUNT; i++) {
      updateFinger(i, now);
    }
  }
}
