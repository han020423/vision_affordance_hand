const int FINGER_COUNT = 4;

// 0: 검지
// 1: 중지
// 2: 약지·소지
// 3: 엄지
const int MOTOR_IN1[FINGER_COUNT] = {2, 4, 6, 8};
const int MOTOR_IN2[FINGER_COUNT] = {3, 5, 7, 9};

const int FEEDBACK_PIN[FINGER_COUNT] = {
  A0, A1, A2, A3
};

// 한 번 명령할 때 작동 시간
const int PULSE_TIME_MS = 100;

void stopMotor(int motor) {
  digitalWrite(MOTOR_IN1[motor], LOW);
  digitalWrite(MOTOR_IN2[motor], LOW);
}

void stopAllMotors() {
  for (int i = 0; i < FINGER_COUNT; i++) {
    stopMotor(i);
  }
}

void direction1(int motor) {
  digitalWrite(MOTOR_IN1[motor], HIGH);
  digitalWrite(MOTOR_IN2[motor], LOW);
}

void direction2(int motor) {
  digitalWrite(MOTOR_IN1[motor], LOW);
  digitalWrite(MOTOR_IN2[motor], HIGH);
}

void movePulse(int motor, int direction) {
  if (direction == 1) {
    direction1(motor);
  }
  else {
    direction2(motor);
  }

  delay(PULSE_TIME_MS);
  stopMotor(motor);
}

int readPosition(int motor) {
  return analogRead(FEEDBACK_PIN[motor]);
}

void printMotorPosition(int motor) {
  switch (motor) {
    case 0:
      Serial.print("검지 A0 위치값: ");
      break;

    case 1:
      Serial.print("중지 A1 위치값: ");
      break;

    case 2:
      Serial.print("약지·소지 A2 위치값: ");
      break;

    case 3:
      Serial.print("엄지 A3 위치값: ");
      break;
  }

  Serial.println(readPosition(motor));
}

void printAllPositions() {
  Serial.println();
  Serial.println("----- 전체 위치값 -----");

  Serial.print("검지 A0: ");
  Serial.println(readPosition(0));

  Serial.print("중지 A1: ");
  Serial.println(readPosition(1));

  Serial.print("약지·소지 A2: ");
  Serial.println(readPosition(2));

  Serial.print("엄지 A3: ");
  Serial.println(readPosition(3));

  Serial.println("----------------------");
}

void runCommand(int motor, int direction) {
  movePulse(motor, direction);

  switch (motor) {
    case 0:
      Serial.print("검지");
      break;

    case 1:
      Serial.print("중지");
      break;

    case 2:
      Serial.print("약지·소지");
      break;

    case 3:
      Serial.print("엄지");
      break;
  }

  Serial.print(" 방향 ");
  Serial.print(direction);
  Serial.println("로 이동");

  printMotorPosition(motor);
}

void printCommands() {
  Serial.println();
  Serial.println("PQ12 Brunel Hand 테스트");
  Serial.println("----------------------");
  Serial.println("1 : 검지 방향 1");
  Serial.println("2 : 검지 방향 2");
  Serial.println("3 : 중지 방향 1");
  Serial.println("4 : 중지 방향 2");
  Serial.println("5 : 약지·소지 방향 1");
  Serial.println("6 : 약지·소지 방향 2");
  Serial.println("7 : 엄지 방향 1");
  Serial.println("8 : 엄지 방향 2");
  Serial.println("s : 모든 모터 정지");
  Serial.println("p : 모든 위치값 출력");
  Serial.println("----------------------");
}

void setup() {
  Serial.begin(115200);

  // Nano 33 IoT ADC를 12비트로 설정
  analogReadResolution(12);

  for (int i = 0; i < FINGER_COUNT; i++) {
    pinMode(MOTOR_IN1[i], OUTPUT);
    pinMode(MOTOR_IN2[i], OUTPUT);
    pinMode(FEEDBACK_PIN[i], INPUT);
  }

  stopAllMotors();

  delay(1000);

  printCommands();
  printAllPositions();
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case '1':
        runCommand(0, 1);
        break;

      case '2':
        runCommand(0, 2);
        break;

      case '3':
        runCommand(1, 1);
        break;

      case '4':
        runCommand(1, 2);
        break;

      case '5':
        runCommand(2, 1);
        break;

      case '6':
        runCommand(2, 2);
        break;

      case '7':
        runCommand(3, 1);
        break;

      case '8':
        runCommand(3, 2);
        break;

      case 's':
      case 'S':
        stopAllMotors();
        Serial.println("모든 모터 정지");
        printAllPositions();
        break;

      case 'p':
      case 'P':
        printAllPositions();
        break;
    }
  }
}