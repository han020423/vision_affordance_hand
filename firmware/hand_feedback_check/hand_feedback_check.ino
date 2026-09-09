// PQ12 위치 피드백 진단 스케치 (모터 구동 없음)
//
// 목적: A0~A3(검지, 중지, 약지·소지, 엄지)의 ADC 값을 주기적으로 출력해
//       배선 점검(3V3 공급, wiper 연결, 납땜 접촉)과 잡음 폭 측정에 쓴다.
// 안전: 모터 IN 핀은 모두 OUTPUT LOW(coast)로 고정하고 절대 바꾸지 않는다.
//       10 V 모터 전원은 꺼 둔 채 USB만 연결해도 동작한다.
//
// 시리얼 115200. 명령:
//   h : 도움말
//   r : 최소/최대 기록 초기화
//   + : 출력 주기 짧게 (최소 50 ms)
//   - : 출력 주기 길게 (최대 2000 ms)
//   p : 지금 즉시 1회 출력
//
// 출력 한 줄 형식 (채널당 "원시값 / 8회 평균 [최소~최대]"):
//   A0 3423/3424 [3421~3427] | A1 ...
// 정상이면 각 채널 값이 손가락 위치에 따라 수백~삼천대에서 움직이고,
// 정지 상태에서는 최소~최대 폭이 수 count 이내여야 한다. 0~20 근처에 머물면
// 3V3 공급(PQ12 핀 4) 또는 wiper(핀 5) 배선을 의심한다.

const int FINGER_COUNT = 4;
const int MOTOR_IN1[FINGER_COUNT] = {2, 4, 6, 8};
const int MOTOR_IN2[FINGER_COUNT] = {3, 5, 7, 9};
const int FEEDBACK_PIN[FINGER_COUNT] = {A0, A1, A2, A3};
const char* FINGER_NAME[FINGER_COUNT] = {"검지", "중지", "약지·소지", "엄지"};

const int AVERAGE_SAMPLES = 8;
unsigned long printIntervalMs = 200;
unsigned long lastPrintMs = 0;

int minValue[FINGER_COUNT];
int maxValue[FINGER_COUNT];

void resetMinMax() {
  for (int i = 0; i < FINGER_COUNT; i++) {
    minValue[i] = 4095;
    maxValue[i] = 0;
  }
}

int readAverage(int pin) {
  long sum = 0;
  for (int k = 0; k < AVERAGE_SAMPLES; k++) {
    sum += analogRead(pin);
  }
  return (int)(sum / AVERAGE_SAMPLES);
}

void printHelp() {
  Serial.println();
  Serial.println("PQ12 피드백 진단 (모터 구동 없음)");
  Serial.println("h 도움말 | r 최소/최대 초기화 | + 빠르게 | - 느리게 | p 즉시 출력");
  Serial.print("출력 주기: ");
  Serial.print(printIntervalMs);
  Serial.println(" ms");
  Serial.println("채널당 표시: 원시값/평균 [최소~최대]");
}

void printReadings() {
  for (int i = 0; i < FINGER_COUNT; i++) {
    int raw = analogRead(FEEDBACK_PIN[i]);
    int avg = readAverage(FEEDBACK_PIN[i]);
    if (avg < minValue[i]) minValue[i] = avg;
    if (avg > maxValue[i]) maxValue[i] = avg;
    Serial.print("A");
    Serial.print(i);
    Serial.print(" ");
    Serial.print(raw);
    Serial.print("/");
    Serial.print(avg);
    Serial.print(" [");
    Serial.print(minValue[i]);
    Serial.print("~");
    Serial.print(maxValue[i]);
    Serial.print("]");
    if (i < FINGER_COUNT - 1) Serial.print(" | ");
  }
  Serial.println();
}

void setup() {
  // 모터 드라이버 입력을 먼저 LOW로 고정한다 (coast). 이 스케치는 절대 HIGH로 바꾸지 않는다.
  for (int i = 0; i < FINGER_COUNT; i++) {
    pinMode(MOTOR_IN1[i], OUTPUT);
    pinMode(MOTOR_IN2[i], OUTPUT);
    digitalWrite(MOTOR_IN1[i], LOW);
    digitalWrite(MOTOR_IN2[i], LOW);
    pinMode(FEEDBACK_PIN[i], INPUT);
  }
  analogReadResolution(12);
  resetMinMax();
  Serial.begin(115200);
  delay(1000);
  printHelp();
  Serial.print("손가락 순서: ");
  for (int i = 0; i < FINGER_COUNT; i++) {
    Serial.print("A");
    Serial.print(i);
    Serial.print("=");
    Serial.print(FINGER_NAME[i]);
    if (i < FINGER_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
}

void loop() {
  while (Serial.available() > 0) {
    char command = Serial.read();
    switch (command) {
      case 'h': case 'H':
        printHelp();
        break;
      case 'r': case 'R':
        resetMinMax();
        Serial.println("최소/최대 초기화");
        break;
      case '+':
        if (printIntervalMs > 50) printIntervalMs /= 2;
        if (printIntervalMs < 50) printIntervalMs = 50;
        Serial.print("출력 주기: "); Serial.print(printIntervalMs); Serial.println(" ms");
        break;
      case '-':
        if (printIntervalMs < 2000) printIntervalMs *= 2;
        if (printIntervalMs > 2000) printIntervalMs = 2000;
        Serial.print("출력 주기: "); Serial.print(printIntervalMs); Serial.println(" ms");
        break;
      case 'p': case 'P':
        printReadings();
        break;
      default:
        break;  // 줄바꿈 등은 무시
    }
  }

  unsigned long now = millis();
  if (now - lastPrintMs >= printIntervalMs) {
    lastPrintMs = now;
    printReadings();
  }
}
