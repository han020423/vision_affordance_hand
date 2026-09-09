// Brunel Hand 폐루프 펌웨어 자료형
// (Arduino 빌더가 .ino 안의 struct/enum보다 함수 원형을 먼저 삽입하므로 별도 헤더에 둔다)

#pragma once

enum FingerState {
  ST_UNCAL = 0,   // 보정값 없음: MOVE 거부, PULSE만 허용
  ST_IDLE,        // 정지
  ST_WAIT,        // 휴지 시간 후 이동 시작 대기
  ST_MOVING,      // 폐루프 이동 중
  ST_PULSE,       // 시간 기반 펄스 구동 중
  ST_DONE,        // 목표 도달
  ST_STALL,       // 진행 없음으로 정지
  ST_TIMEOUT,     // 최대 구동시간 초과로 정지
  ST_FAULT        // 센서 범위 이탈로 정지
};

struct Finger {
  FingerState state;
  bool calibrated;
  int softLow;
  int softHigh;
  int hardLow;
  int hardHigh;
  int increaseDir;        // 1 또는 2: 이 방향 번호로 구동하면 ADC 증가
  bool extendedIsHigh;    // 펼친 상태가 높은 ADC인지
  unsigned long maxMoveMs;

  int target;             // 목표 ADC
  int dir;                // +1: ADC 증가 쪽, -1: 감소 쪽, 0: 정지
  int reversals;          // 오버슛 보정으로 방향을 바꾼 횟수
  unsigned long moveStartMs;
  unsigned long lastProgressMs;
  int lastProgressAdc;
  unsigned long restUntilMs;
  unsigned long pulseEndMs;
  int lastAdc;            // 마지막으로 읽은 평균 ADC
};
