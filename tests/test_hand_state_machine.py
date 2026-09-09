"""비전 → 손 상태 머신 시험 (MockLink, 하드웨어 없이).

프리셋 비율은 아직 실측되지 않았으므로, 이 시험에서는 **시험용 임시 YAML**에
비율을 채워 상태 전이를 검증한다. 실제 `configs/hardware/hand_actuators.yaml`은
건드리지 않으며, 여기의 값은 하드웨어 보정값이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.hand_control import HandClient, MockLink, load_calibration  # noqa: E402
from src.hand_control.state_machine import (  # noqa: E402
    ALIGN,
    CLOSE,
    FAULT,
    HOLD,
    OPEN,
    PRE_SHAPE,
    SEARCH,
    TARGET_SELECTED,
    GraspController,
    candidate_key,
)

REAL_CONFIG = REPOSITORY_ROOT / "configs" / "hardware" / "hand_actuators.yaml"

# 시험용 프리셋 블록. 파지 비율과 순서는 실제 설정과 같게 두고, PRE_SHAPE만
# 0이 아닌 값을 써서 반개방 단계가 실제로 움직이는지 확인할 수 있게 한다
# (실제 설정의 PRE_SHAPE는 아직 전부 0이라 이동이 없다).
TEST_PRESETS = """
presets:
  OPEN: [0.0, 0.0, 0.0, 0.0]
  PRECISION: [1.0, 1.0, 0.0, 1.0]
  WRAP: [1.0, 1.0, 1.0, 0.0]
  POWER: [1.0, 1.0, 1.0, 1.0]
  PRE_SHAPE:
    PRECISION: [0.30, 0.30, 0.0, 0.25]
    WRAP: [0.35, 0.35, 0.35, 0.0]
    POWER: [0.40, 0.40, 0.40, 0.30]

# 실제 설정과 같은 구동 순서(엄지 받침 먼저)
preset_sequence:
  OPEN: [[0, 1, 2, 3]]
  PRECISION: [[2, 3], [0, 1]]
  WRAP: [[3], [0, 1, 2]]
  POWER: [[3], [0, 1, 2]]
"""


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


@dataclass
class FakeCandidate:
    """grasp_selection.GraspCandidate에서 상태 머신이 쓰는 부분만 흉내낸다."""

    class_id: int = 0
    grasp_point: tuple[int, int] = (100, 100)


@dataclass
class FakeDecision:
    """grasp_selection.Decision의 최소 형태."""

    state: str = "GRASP"
    pose: str | None = "WRAP"
    candidate: FakeCandidate | None = field(default_factory=FakeCandidate)
    reason: str = ""


def write_test_config(directory: str) -> Path:
    """실제 보정값 + 시험용 프리셋으로 임시 설정 파일을 만든다."""

    text = REAL_CONFIG.read_text(encoding="utf-8")
    head = text.split("# 프리셋:", 1)[0]
    path = Path(directory) / "hand_actuators_test.yaml"
    path.write_text(head + TEST_PRESETS, encoding="utf-8")
    return path


class StateMachineFixture(unittest.TestCase):
    """공용 준비 코드."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.config_path = write_test_config(self._temporary.name)
        self.calibration = load_calibration(self.config_path)
        self.clock = FakeClock()
        self.link = MockLink(
            self.calibration,
            clock=self.clock,
            advance=self.clock.advance,
            initial=[3330, 3400, 3420, 420],   # 모두 펼친 상태
        )
        self.hand = HandClient(self.link, self.calibration, clock=self.clock)
        self.hand.connect()
        self.controller = GraspController(
            self.hand, self.calibration, stable_frames=3, clock=self.clock
        )

    def advance_until_idle(self, decision, *, hand_valid: bool = True, limit: int = 400):
        """진행 중 이동(pending)이 없어질 때까지 프레임을 돌리고 마지막 상태를 돌려준다.

        이동이 끝나는 프레임의 update()가 이어서 상태 전이까지 수행하므로
        (예: CLOSE 이동 완료 → HOLD) 반환값은 전이 후 상태다.
        """

        status = self.controller.update(decision, hand_valid=hand_valid)
        for _ in range(limit):
            if not status.pending:
                return status
            self.clock.advance(0.05)
            status = self.controller.update(decision, hand_valid=hand_valid)
        self.fail("이동이 끝나지 않았습니다(상한 초과)")


class TestCandidateKey(unittest.TestCase):
    def test_small_jitter_is_same_candidate(self):
        first = candidate_key(FakeCandidate(0, (100, 100)))
        second = candidate_key(FakeCandidate(0, (110, 105)))
        self.assertEqual(first, second)

    def test_large_move_is_different_candidate(self):
        first = candidate_key(FakeCandidate(0, (100, 100)))
        second = candidate_key(FakeCandidate(0, (400, 100)))
        self.assertNotEqual(first, second)

    def test_class_change_is_different_candidate(self):
        self.assertNotEqual(
            candidate_key(FakeCandidate(0, (100, 100))),
            candidate_key(FakeCandidate(1, (100, 100))),
        )

    def test_none_candidate(self):
        self.assertIsNone(candidate_key(None))


class TestGraspSequence(StateMachineFixture):
    def test_full_sequence_requires_user_trigger(self):
        decision = FakeDecision()

        # 안정화 프레임이 쌓이기 전에는 SEARCH에 머문다.
        self.assertEqual(self.controller.update(decision).state, SEARCH)
        self.assertEqual(self.controller.update(decision).state, SEARCH)
        self.assertEqual(self.controller.update(decision).state, TARGET_SELECTED)

        # TARGET_SELECTED → PRE_SHAPE(반개방) 이동 시작 후 완료까지.
        # 트리거가 없으므로 완료 후에도 PRE_SHAPE에서 대기한다.
        status = self.advance_until_idle(decision)
        self.assertEqual(status.state, PRE_SHAPE)
        self.assertIn("파지 대기", status.detail)

        # 트리거 후 CLOSE 이동 → 완료와 함께 HOLD
        self.controller.request_close()
        status = self.advance_until_idle(decision)
        self.assertEqual(status.state, HOLD)

        # 파지 후 위치가 WRAP 프리셋 목표에 도달했는지 확인
        positions = self.hand.positions()
        for finger, fraction in zip(self.calibration.fingers, self.calibration.preset("WRAP")):
            expected = finger.fraction_to_adc(fraction)
            self.assertLessEqual(
                abs(positions[finger.index] - expected), self.calibration.deadband_adc
            )

    def test_auto_close_skips_trigger(self):
        controller = GraspController(
            self.hand, self.calibration, stable_frames=1, auto_close=True, clock=self.clock
        )
        decision = FakeDecision(pose="POWER")
        self.assertEqual(controller.update(decision).state, TARGET_SELECTED)
        self.controller = controller
        # 트리거 없이도 PRE_SHAPE 완료 → CLOSE → HOLD까지 한 번에 이어진다.
        self.assertEqual(self.advance_until_idle(decision).state, HOLD)

    def test_blocked_finger_is_reported_as_stall_in_hold(self):
        self.link.blocked = {1}
        decision = FakeDecision(pose="POWER")
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)
        self.controller.request_close()
        status = self.advance_until_idle(decision)
        self.assertEqual(status.state, HOLD)
        self.assertIn("막힘", status.detail)
        self.assertTrue(status.last_results[1].blocked)

    def test_target_lost_before_close_returns_to_open(self):
        decision = FakeDecision()
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)   # PRE_SHAPE 도달

        lost = FakeDecision(state="NO_TARGET", pose=None, candidate=None)
        status = self.controller.update(lost)
        self.assertEqual(status.state, OPEN)          # 정지 후 펼침 시작
        # 펼침 완료와 함께 SEARCH로 돌아간다.
        self.assertEqual(self.advance_until_idle(lost).state, SEARCH)

    def test_hand_tracking_lost_before_close_returns_to_open(self):
        decision = FakeDecision()
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)
        status = self.controller.update(decision, hand_valid=False)
        self.assertEqual(status.state, OPEN)
        self.assertIn("손 추적 소실", status.detail)

    def test_align_state_does_not_move(self):
        before = self.hand.positions()
        align = FakeDecision(state="ALIGN", pose=None, candidate=None, reason="1·2위 점수 차 0.03")
        for _ in range(5):
            status = self.controller.update(align)
        self.assertEqual(status.state, ALIGN)
        self.assertEqual(self.hand.positions(), before)

    def test_candidate_switch_resets_stability(self):
        first = FakeDecision(candidate=FakeCandidate(0, (100, 100)))
        second = FakeDecision(candidate=FakeCandidate(1, (500, 300)))
        self.controller.update(first)
        self.controller.update(first)
        status = self.controller.update(second)   # 후보가 바뀌면 다시 1부터
        self.assertEqual(status.state, SEARCH)
        self.assertIn("1/3", status.detail)

    def test_emergency_stop_from_hold(self):
        decision = FakeDecision()
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)
        self.controller.request_close()
        self.assertEqual(self.advance_until_idle(decision).state, HOLD)
        self.controller.emergency_stop()
        self.assertEqual(self.controller.status.state, SEARCH)
        self.assertIn("비상정지", self.controller.status.detail)

    def test_open_request_from_hold(self):
        decision = FakeDecision()
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)
        self.controller.request_close()
        self.assertEqual(self.advance_until_idle(decision).state, HOLD)
        self.controller.request_open()
        self.assertEqual(self.controller.update(decision).state, OPEN)
        self.assertEqual(self.advance_until_idle(decision).state, SEARCH)
        positions = self.hand.positions()
        for finger in self.calibration.fingers:
            self.assertLessEqual(
                abs(positions[finger.index] - finger.open_adc), self.calibration.deadband_adc
            )

    def test_heartbeat_stop_moves_to_fault(self):
        """이동 중 펌웨어가 heartbeat로 멈추면 FAULT로 남아 사람이 개입하게 한다."""

        decision = FakeDecision()
        for _ in range(3):
            self.controller.update(decision)
        self.controller.update(decision)          # PRE_SHAPE 이동 시작
        self.link._emit("EVT HEARTBEAT_STOP")
        status = self.controller.update(decision)
        self.assertEqual(status.state, FAULT)


class TestUndefinedPresets(unittest.TestCase):
    """프리셋 비율이 정해지지 않았으면 추측해서 구동하지 않고 알려야 한다."""

    def _config_without_pre_shape(self, directory: str) -> Path:
        text = REAL_CONFIG.read_text(encoding="utf-8")
        head = text.split("# 프리셋:", 1)[0]
        path = Path(directory) / "no_pre_shape.yaml"
        path.write_text(head + """
presets:
  OPEN: [0.0, 0.0, 0.0, 0.0]
  WRAP: [1.0, 1.0, 1.0, 0.0]
  PRE_SHAPE:
    WRAP: null
""", encoding="utf-8")
        return path

    def test_missing_pre_shape_does_not_move_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = load_calibration(self._config_without_pre_shape(directory))
            clock = FakeClock()
            link = MockLink(calibration, clock=clock, advance=clock.advance)
            hand = HandClient(link, calibration, clock=clock)
            hand.connect()
            controller = GraspController(hand, calibration, stable_frames=1, clock=clock)
            before = hand.positions()
            self.assertEqual(controller.update(FakeDecision(pose="WRAP")).state, TARGET_SELECTED)
            status = controller.update(FakeDecision(pose="WRAP"))
            self.assertEqual(status.state, TARGET_SELECTED)
            self.assertIn("PRE_SHAPE 미정", status.detail)
            self.assertEqual(hand.positions(), before)   # 손은 움직이지 않았다

    def test_real_config_reaches_close_with_user_presets(self):
        """실제 설정(사용자 정의 프리셋)으로는 CLOSE까지 진행되어야 한다."""

        calibration = load_calibration(REAL_CONFIG)
        clock = FakeClock()
        link = MockLink(calibration, clock=clock, advance=clock.advance,
                        initial=[3330, 3400, 3420, 420])
        hand = HandClient(link, calibration, clock=clock)
        hand.connect()
        controller = GraspController(hand, calibration, stable_frames=1,
                                     auto_close=True, clock=clock)
        decision = FakeDecision(pose="WRAP")
        status = controller.update(decision)
        self.assertEqual(status.state, TARGET_SELECTED)
        for _ in range(400):
            status = controller.update(decision)
            if status.state == HOLD:
                break
            clock.advance(0.05)
        self.assertEqual(status.state, HOLD)
        # WRAP은 엄지를 쓰지 않으므로 엄지는 펼친 위치에 남아야 한다.
        positions = hand.positions()
        thumb = calibration.finger(3)
        self.assertLessEqual(abs(positions[3] - thumb.open_adc), calibration.deadband_adc)
        for index in (0, 1, 2):
            finger = calibration.finger(index)
            self.assertLessEqual(
                abs(positions[index] - finger.closed_adc), calibration.deadband_adc
            )


if __name__ == "__main__":
    unittest.main()


class TestSequencedClose(StateMachineFixture):
    """CLOSE가 프리셋 순서를 단계별로 진행하는지 확인한다."""

    def test_close_runs_two_steps_for_power(self):
        decision = FakeDecision(pose="POWER")
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)            # PRE_SHAPE 완료
        self.controller.request_close()

        status = self.controller.update(decision)    # 1단계 시작
        self.assertEqual(status.state, CLOSE)
        self.assertIn("1/2단계", status.detail)
        self.assertEqual(status.pending, (3,))       # 엄지만

        # 1단계 완료 → 2단계 자동 시작
        for _ in range(400):
            self.clock.advance(0.05)
            status = self.controller.update(decision)
            if "2/2단계" in status.detail:
                break
        self.assertEqual(status.state, CLOSE)
        self.assertEqual(status.pending, (0, 1, 2))  # 나머지 세 손가락

        # 2단계 완료 → HOLD
        status = self.advance_until_idle(decision)
        self.assertEqual(status.state, HOLD)

    def test_wrap_also_moves_thumb_first(self):
        """WRAP은 엄지를 쓰지 않지만, 먼저 비켜두고 나머지를 접는다."""

        decision = FakeDecision(pose="WRAP")
        for _ in range(3):
            self.controller.update(decision)
        self.advance_until_idle(decision)
        self.controller.request_close()
        status = self.controller.update(decision)
        self.assertEqual(status.pending, (3,))
        # 1단계(엄지) → 2단계(나머지) → HOLD
        for _ in range(400):
            self.clock.advance(0.05)
            status = self.controller.update(decision)
            if status.state == HOLD:
                break
        self.assertEqual(status.state, HOLD)
        positions = self.hand.positions()
        thumb = self.calibration.finger(3)
        self.assertLessEqual(abs(positions[3] - thumb.open_adc), self.calibration.deadband_adc)
