"""비전 결과(Decision)를 손 동작으로 옮기는 상태 머신.

`PROJECT_CONTEXT.md` 12절의 상태 머신을 구현한다.

    SEARCH → TARGET_SELECTED → PRE_SHAPE → CLOSE → HOLD → OPEN → SEARCH
                    ↓ (후보 불안정)
                  ALIGN

설계 원칙

- **카메라 루프를 막지 않는다.** `update()`는 프레임마다 한 번 호출되며 절대
  블로킹 대기를 하지 않는다. 이동 완료는 다음 프레임들에서 이벤트로 확인한다.
- **파지(CLOSE)는 기본적으로 사용자 트리거**로만 시작한다(`auto_close=True`면 자동).
  `AGENTS.md` 8절의 "사용자가 확인한 뒤 구동" 원칙을 따른다.
- CLOSE 전에 후보가 사라지거나(`ALIGN`/`NO_TARGET`) 손 추적이 끊기면 정지하고
  펼침으로 되돌린다.
- 이동 중에는 `ping()`으로 펌웨어 heartbeat를 유지한다. 프로세스가 죽으면
  펌웨어가 스스로 멈춘다.
- 프리셋 비율이 아직 정해지지 않았으면(`PRECISION`/`WRAP`/`POWER`가 null)
  추측해서 구동하지 않고 상태 텍스트로 알린다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .client import HandClient, HandError, MoveResult
from .presets import CalibrationError, HandCalibration

# 상태 이름 (PROJECT_CONTEXT.md 12절)
SEARCH = "SEARCH"
TARGET_SELECTED = "TARGET_SELECTED"
ALIGN = "ALIGN"
PRE_SHAPE = "PRE_SHAPE"
CLOSE = "CLOSE"
HOLD = "HOLD"
OPEN = "OPEN"
FAULT = "FAULT"


@dataclass
class ControllerStatus:
    """화면 표시와 로그용 요약."""

    state: str
    pose: str | None = None
    detail: str = ""
    pending: tuple[int, ...] = ()
    last_results: dict[int, MoveResult] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.state]
        if self.pose:
            parts.append(self.pose)
        if self.detail:
            parts.append(self.detail)
        return " | ".join(parts)


def candidate_key(candidate, tolerance_px: int = 40) -> tuple[int, int, int] | None:
    """같은 후보인지 비교할 키. 파지점이 tolerance 안에서 흔들리면 같다고 본다."""

    if candidate is None:
        return None
    x, y = candidate.grasp_point
    return (int(candidate.class_id), int(x) // tolerance_px, int(y) // tolerance_px)


class GraspController:
    """Decision 흐름을 받아 HandClient를 구동한다.

    사용법(카메라 루프 안):

        controller.update(decision, hand_valid=hand_state is not None)
        overlay_text = controller.status.text

    키 입력은 `request_close()`, `request_open()`, `emergency_stop()`으로 전달한다.
    """

    def __init__(
        self,
        hand: HandClient,
        calibration: HandCalibration,
        *,
        stable_frames: int = 5,
        auto_close: bool = False,
        move_timeout: float = 6.0,
        ping_interval: float = 0.3,
        log=None,
        clock=time.monotonic,
    ) -> None:
        self._hand = hand
        self._cal = calibration
        self._stable_frames = max(1, stable_frames)
        self._auto_close = auto_close
        self._move_timeout = move_timeout
        self._ping_interval = ping_interval
        self._log = log
        self._clock = clock

        self.status = ControllerStatus(state=SEARCH)
        self._pose: str | None = None
        self._candidate_key: tuple[int, int, int] | None = None
        self._stable_count = 0
        self._pending: set[int] = set()
        self._move_started = 0.0
        self._last_ping = 0.0
        self._close_requested = False
        self._open_requested = False
        self._preset_name = ""
        self._preset_step = 0
        self._preset_total = 0

    # ------------------------------------------------------------------ 도우미

    def _note(self, message: str) -> None:
        if self._log is not None:
            self._log(f"CTRL {self.status.state} {message}")

    def _set_state(self, state: str, detail: str = "") -> None:
        if state != self.status.state:
            self._note(f"-> {state} {detail}".strip())
        self.status = ControllerStatus(
            state=state,
            pose=self._pose,
            detail=detail,
            pending=tuple(sorted(self._pending)),
            last_results=self.status.last_results,
        )

    def _start_preset(self, name: str) -> bool:
        """프리셋 구동을 첫 단계부터 시작한다. 프리셋이 미정이면 False.

        물체를 쥘 때는 순서가 중요하므로(엄지 받침 먼저) 단계별로 진행한다.
        """

        try:
            self._preset_total = len(self._cal.preset_steps(name))
        except CalibrationError as error:
            self._set_state(self.status.state, f"프리셋 미정: {error}")
            return False
        self._preset_name = name
        self._preset_step = 0
        return self._start_preset_step()

    def _start_preset_step(self) -> bool:
        """현재 단계를 구동한다."""

        try:
            group = self._hand.move_preset_step(self._preset_name, self._preset_step)
        except CalibrationError as error:
            self._set_state(self.status.state, f"프리셋 미정: {error}")
            return False
        except HandError as error:
            self._set_state(FAULT, f"명령 실패: {error}")
            return False
        self._pending = set(group)
        self._move_started = self._clock()
        self._last_ping = self._clock()
        return True

    def _start_open(self) -> bool:
        try:
            self._hand.open()
        except HandError as error:
            self._set_state(FAULT, f"펼침 실패: {error}")
            return False
        self._pending = set(range(len(self._cal.fingers)))
        self._move_started = self._clock()
        self._last_ping = self._clock()
        return True

    def _service_pending(self) -> bool:
        """진행 중 이동을 점검한다. 아직 진행 중이면 True.

        블로킹하지 않고 이벤트만 확인하며, heartbeat 유지를 위해 PING을 보낸다.
        """

        if not self._pending:
            return False
        results = dict(self.status.last_results)
        for event in self._hand.poll_events(timeout=0.0):
            if event.is_settle and event.finger in self._pending:
                self._pending.discard(event.finger)
                results[event.finger] = MoveResult(event.finger, event.kind, event.adc)
            elif event.kind == "HEARTBEAT_STOP":
                self._pending.clear()
                self.status.last_results = results
                self._set_state(FAULT, "펌웨어 heartbeat 정지")
                return False
        self.status.last_results = results

        now = self._clock()
        if self._pending and now - self._last_ping >= self._ping_interval:
            try:
                self._hand.ping()
            except HandError:
                pass
            self._last_ping = now

        if self._pending and now - self._move_started > self._move_timeout:
            try:
                self._hand.stop()
            except HandError:
                pass
            self._note(f"이동 시간 초과, 정지 (남은 손가락 {sorted(self._pending)})")
            self._pending.clear()
            return False
        return bool(self._pending)

    def _abort_to_open(self, reason: str) -> None:
        """CLOSE 전 이상 상황: 정지하고 펼침으로 되돌린다."""

        try:
            self._hand.stop()
        except HandError:
            pass
        self._pending.clear()
        self._pose = None
        self._candidate_key = None
        self._stable_count = 0
        self._close_requested = False
        self._start_open()
        self._set_state(OPEN, reason)

    # ------------------------------------------------------------------ 외부 입력

    def request_close(self) -> None:
        """사용자 파지 트리거(예: 스페이스 키)."""

        self._close_requested = True

    def request_open(self) -> None:
        """사용자 펼침 트리거(예: o 키)."""

        self._open_requested = True

    def emergency_stop(self) -> None:
        """비상정지(예: x 키). 어떤 상태에서도 즉시 정지한다."""

        try:
            self._hand.stop()
        except HandError:
            pass
        self._pending.clear()
        self._pose = None
        self._candidate_key = None
        self._stable_count = 0
        self._close_requested = False
        self._open_requested = False
        self._set_state(SEARCH, "비상정지")

    # ------------------------------------------------------------------ 본체

    def update(self, decision, *, hand_valid: bool = True) -> ControllerStatus:
        """프레임 하나를 처리한다. decision은 grasp_selection의 Decision."""

        state = self.status.state

        # 사용자 펼침 요청은 어느 상태에서나 우선 처리한다.
        if self._open_requested:
            self._open_requested = False
            self._pose = None
            self._candidate_key = None
            self._stable_count = 0
            self._close_requested = False
            self._start_open()
            self._set_state(OPEN, "사용자 요청")
            return self.status

        if state == FAULT:
            return self.status

        decision_state = getattr(decision, "state", "NO_TARGET") if decision is not None else "NO_TARGET"
        pose = getattr(decision, "pose", None) if decision is not None else None
        key = candidate_key(getattr(decision, "candidate", None)) if decision is not None else None

        # 파지 전 단계에서 대상이 사라지거나 손 추적이 끊기면 되돌린다.
        # 반개방 이동이 진행 중이어도 즉시 중단해야 하므로 pending 점검보다 먼저 본다.
        if state in {TARGET_SELECTED, PRE_SHAPE} and (decision_state != "GRASP" or not hand_valid):
            reason = "손 추적 소실" if not hand_valid else f"후보 불안정({decision_state})"
            self._abort_to_open(reason)
            return self.status

        # 이동 중이면 완료를 기다린다(비차단).
        if self._service_pending():
            self._set_state(state, f"이동 중 {sorted(self._pending)}")
            return self.status

        # 이동 점검 중 상태가 바뀌었을 수 있다(예: heartbeat 정지 → FAULT).
        # 위에서 읽어둔 state로 계속 진행하면 그 전이를 덮어쓰므로 다시 읽는다.
        state = self.status.state
        if state == FAULT:
            return self.status

        if state == OPEN:
            # 펼침 완료 → 다시 탐색
            self._set_state(SEARCH, "")
            return self.status

        if state == SEARCH or state == ALIGN:
            if not hand_valid:
                self._set_state(SEARCH, "손 미검출")
                return self.status
            if decision_state == "GRASP" and pose:
                if key == self._candidate_key and pose == self._pose:
                    self._stable_count += 1
                else:
                    self._candidate_key = key
                    self._pose = pose
                    self._stable_count = 1
                if self._stable_count >= self._stable_frames:
                    self._set_state(TARGET_SELECTED, f"{self._stable_count}프레임 안정")
                else:
                    self._set_state(
                        SEARCH, f"{pose} 안정화 {self._stable_count}/{self._stable_frames}"
                    )
            elif decision_state == "ALIGN":
                self._stable_count = 0
                self._set_state(ALIGN, getattr(decision, "reason", ""))
            else:
                self._stable_count = 0
                self._pose = None
                self._candidate_key = None
                self._set_state(SEARCH, "대상 없음")
            return self.status

        if state == TARGET_SELECTED:
            # 자세별 반개방 자세로 미리 벌린다. 미정이면 그대로 대기.
            if self._pose and self._start_pre_shape(self._pose):
                self._set_state(PRE_SHAPE, f"{self._pose} 반개방")
            return self.status

        if state == PRE_SHAPE:
            if not (self._close_requested or self._auto_close):
                self._set_state(PRE_SHAPE, "파지 대기(스페이스)")
                return self.status
            self._close_requested = False
            if self._pose and self._start_preset(self._pose):
                self._set_state(CLOSE, f"{self._pose} 1/{self._preset_total}단계")
            return self.status

        if state == CLOSE:
            # 한 단계가 끝났다. 남은 단계가 있으면 이어서 구동한다.
            self._preset_step += 1
            if self._preset_step < self._preset_total:
                if self._start_preset_step():
                    self._set_state(
                        CLOSE, f"{self._preset_name} {self._preset_step + 1}/{self._preset_total}단계"
                    )
                return self.status
            # 스톨은 "무언가에 걸려 멈췄다"는 뜻이고, 목표 도달(DONE)도 얇은 물체나
            # 구조상 더 굽지 않는 자세에서는 파지일 수 있다. 그래서 성공 판정 대신
            # 관측한 사실만 적는다.
            blocked = [index for index, result in self.status.last_results.items() if result.blocked]
            detail = f"막힘 {sorted(blocked)}" if blocked else "목표 도달"
            self._set_state(HOLD, detail)
            return self.status

        if state == HOLD:
            self._set_state(HOLD, "유지 중(o=펼침, x=정지)")
            return self.status

        return self.status

    def _start_pre_shape(self, pose: str) -> bool:
        try:
            self._hand.move_pre_shape(pose)
        except CalibrationError as error:
            self._set_state(TARGET_SELECTED, f"PRE_SHAPE 미정: {error}")
            return False
        except HandError as error:
            self._set_state(FAULT, f"명령 실패: {error}")
            return False
        self._pending = set(range(len(self._cal.fingers)))
        self._move_started = self._clock()
        self._last_ping = self._clock()
        return True
