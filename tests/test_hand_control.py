"""손 제어 패키지 단위·통합 시험 (하드웨어 없이 MockLink로 수행).

- 프로토콜 파싱: 실제 펌웨어(v2) 출력 문자열을 그대로 사용한다.
- 보정 로더: 실제 `configs/hardware/hand_actuators.yaml`을 읽고, 실측값과 대조한다.
- 비율→ADC 환산: 2026-08-21 실물 확인값(FRAC 200 → 2754/2824/2840/1026)과 일치해야 한다.
- HandClient: 연결 절차, 이동 완료, 스톨, 안전 거부를 가짜 시계로 결정적으로 검증한다.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.hand_control import (  # noqa: E402
    CalibrationError,
    HandClient,
    HandError,
    MockLink,
    load_calibration,
)
from src.hand_control import protocol  # noqa: E402

CONFIG_PATH = REPOSITORY_ROOT / "configs" / "hardware" / "hand_actuators.yaml"


class FakeClock:
    """시험용 가짜 시계. sleep이 시간을 앞으로 감는다."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def make_mock_client(**mock_kwargs):
    """가짜 시계를 공유하는 MockLink + HandClient 쌍을 만든다."""

    clock = FakeClock()
    calibration = load_calibration(CONFIG_PATH)
    link = MockLink(calibration, clock=clock, advance=clock.advance, **mock_kwargs)
    client = HandClient(link, calibration, clock=clock)
    return clock, calibration, link, client


class TestProtocolParsing(unittest.TestCase):
    """실제 펌웨어 v2가 낸 문자열로 파서를 검증한다."""

    def test_identity(self):
        response = protocol.parse_line(
            "OK ID hand_closed_loop fw 2 proto 1 fingers 4 adcbits 12 uptime 15350"
        )
        identity = protocol.parse_identity(response)
        self.assertEqual(identity["name"], "hand_closed_loop")
        self.assertEqual(identity["fw"], "2")
        self.assertEqual(identity["proto"], "1")
        self.assertEqual(identity["fingers"], "4")

    def test_positions(self):
        response = protocol.parse_line("OK POS 3391 3426 3054 416")
        self.assertEqual(protocol.parse_positions(response), [3391, 3426, 3054, 416])

    def test_status(self):
        response = protocol.parse_line("OK STATUS 0:IDLE:3384:- 1:MOVING:3419:2754 2:IDLE:3059:- 3:IDLE:408:-")
        entries = protocol.parse_status(response)
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0].state, "IDLE")
        self.assertIsNone(entries[0].target)
        self.assertEqual(entries[1].state, "MOVING")
        self.assertEqual(entries[1].target, 2754)
        self.assertEqual(entries[3].adc, 408)

    def test_calibration_echo(self):
        response = protocol.parse_line(
            "OK CAL 0:450:3330:1:1:3000 1:520:3400:2:1:3000 2:520:3420:1:1:3000 3:420:3450:1:0:3500"
        )
        echoed = protocol.parse_calibration_echo(response)
        self.assertEqual(echoed[0], (450, 3330, 1, True, 3000))
        self.assertEqual(echoed[1][2], 2)          # 중지는 ADC 증가 방향이 2
        self.assertFalse(echoed[3][3])             # 엄지는 extended_is_high=False

    def test_calibration_echo_skips_uncal(self):
        response = protocol.parse_line("OK CAL 0:450:3330:1:1:3000 1:UNCAL 2:UNCAL 3:UNCAL")
        echoed = protocol.parse_calibration_echo(response)
        self.assertEqual(list(echoed), [0])

    def test_events(self):
        done = protocol.parse_line("EVT DONE 0 2987")
        self.assertEqual((done.kind, done.finger, done.adc), ("DONE", 0, 2987))
        self.assertTrue(done.is_settle)
        stall = protocol.parse_line("EVT STALL 1 1850")
        self.assertTrue(stall.is_settle)
        heartbeat = protocol.parse_line("EVT HEARTBEAT_STOP")
        self.assertEqual(heartbeat.kind, "HEARTBEAT_STOP")
        self.assertFalse(heartbeat.is_settle)
        self.assertIsNone(heartbeat.finger)

    def test_error_and_unknown(self):
        error = protocol.parse_line("ERR RANGE index")
        self.assertTrue(error.failed)
        self.assertEqual(error.kind, "RANGE")
        self.assertIsNone(protocol.parse_line("   "))
        self.assertIsNone(protocol.parse_line("# 주석"))
        self.assertIsNone(protocol.parse_line("검지 A0 위치값: 3452"))  # 옛 펌웨어 출력

    def test_command_strings(self):
        self.assertEqual(protocol.command_move(0, 3000), "MOVE 0 3000")
        self.assertEqual(protocol.command_fraction_all([200, None, 200, None]), "FRACALL 200 -1 200 -1")
        self.assertEqual(protocol.command_set_legacy(False), "SET LEGACY 0")
        self.assertEqual(
            protocol.command_set_calibration(3, 420, 3450, 1, False, 3500),
            "SET CAL 3 420 3450 1 0 3500",
        )


class TestCalibrationLoading(unittest.TestCase):
    """실제 설정 파일을 읽어 실측값과 대조한다."""

    def setUp(self):
        self.calibration = load_calibration(CONFIG_PATH)

    def test_finger_order_and_values(self):
        names = [finger.name for finger in self.calibration.fingers]
        self.assertEqual(names, ["index", "middle", "ring_little", "thumb"])
        self.assertEqual(self.calibration.finger(1).adc_increase_direction, 2)
        self.assertFalse(self.calibration.finger(3).extended_is_high)

    def test_open_closed_sides(self):
        index = self.calibration.finger(0)
        self.assertEqual(index.open_adc, index.soft_high)     # 검지: 높은 값이 펼침
        thumb = self.calibration.finger(3)
        self.assertEqual(thumb.open_adc, thumb.soft_low)      # 엄지: 낮은 값이 펼침

    def test_fraction_matches_firmware_observation(self):
        """2026-08-21 실물에서 FRAC ... 200 이 낸 목표값과 같아야 한다."""

        observed = {0: 2754, 1: 2824, 2: 2840, 3: 1026}
        for index, expected in observed.items():
            with self.subTest(finger=index):
                self.assertEqual(self.calibration.finger(index).fraction_to_adc(0.2), expected)

    def test_fraction_endpoints(self):
        for finger in self.calibration.fingers:
            self.assertEqual(finger.fraction_to_adc(0.0), finger.open_adc)
            self.assertEqual(finger.fraction_to_adc(1.0), finger.closed_adc)

    def test_preset_scheme(self):
        """2026-08-24 사용자 정의: 참여 손가락만 최대로 접고 나머지는 펼친다."""

        self.assertEqual(self.calibration.preset("OPEN"), (0.0, 0.0, 0.0, 0.0))
        # 순서는 [검지, 중지, 약지·소지, 엄지]
        self.assertEqual(self.calibration.preset("PRECISION"), (1.0, 1.0, 0.0, 1.0))
        self.assertEqual(self.calibration.preset("WRAP"), (1.0, 1.0, 1.0, 0.0))
        self.assertEqual(self.calibration.preset("POWER"), (1.0, 1.0, 1.0, 1.0))
        for pose in ("PRECISION", "WRAP", "POWER"):
            with self.subTest(pose=pose):
                self.assertEqual(len(self.calibration.pre_shape_preset(pose)), 4)

    def test_unknown_preset_name_is_rejected(self):
        with self.assertRaises(CalibrationError):
            self.calibration.preset("없는프리셋")

    def test_null_preset_is_rejected(self):
        """비율이 null인 프리셋은 추측해서 구동하지 않는다."""

        import tempfile

        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "  WRAP:      [1.0, 1.0, 1.0, 0.0]", "  WRAP:      null"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "null_preset.yaml"
            path.write_text(text, encoding="utf-8")
            calibration = load_calibration(path)
            self.assertNotIn("WRAP", calibration.available_presets())
            with self.assertRaises(CalibrationError):
                calibration.preset("WRAP")

    def test_unconfirmed_calibration_is_rejected(self):
        import tempfile

        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "calibration_confirmed: true", "calibration_confirmed: false"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unconfirmed.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(CalibrationError):
                load_calibration(path)

    def test_null_value_is_rejected(self):
        import tempfile

        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "    soft_low: 450", "    soft_low: null"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "null.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(CalibrationError):
                load_calibration(path)


class TestHandClientWithMock(unittest.TestCase):
    """MockLink로 HandClient 동작을 결정적으로 검증한다."""

    def test_connect_sequence(self):
        clock, _, link, client = make_mock_client()
        identity = client.connect()
        self.assertEqual(identity["name"], "hand_closed_loop")
        # 연결 절차의 효과: 단일 문자 명령 차단, heartbeat 켜짐
        self.assertFalse(link.legacy_enabled)
        self.assertGreater(link.heartbeat_ms, 0)

    def test_connect_rejects_wrong_firmware(self):
        clock, calibration, link, client = make_mock_client()

        def wrong_id(text: str) -> None:
            if text.strip().upper() == "ID":
                link._emit("OK ID other_firmware fw 9 proto 1 fingers 4")
            else:
                MockLink.write_line(link, text)

        link.write_line = wrong_id  # type: ignore[assignment]
        with self.assertRaises(HandError):
            client.connect()

    def test_open_then_close_reaches_targets(self):
        clock, calibration, link, client = make_mock_client(
            initial=[3300, 3350, 3400, 450]
        )
        client.connect()
        client.close_hand()
        results = client.wait_settled(timeout=10.0)
        self.assertEqual(len(results), 4)
        for index, result in results.items():
            with self.subTest(finger=index):
                self.assertEqual(result.kind, "DONE")
                finger = calibration.finger(index)
                self.assertLessEqual(abs(result.adc - finger.closed_adc), calibration.deadband_adc)

        client.open()
        results = client.wait_settled(timeout=10.0)
        for index, result in results.items():
            with self.subTest(finger=index, phase="open"):
                self.assertEqual(result.kind, "DONE")
                finger = calibration.finger(index)
                self.assertLessEqual(abs(result.adc - finger.open_adc), calibration.deadband_adc)

    def test_blocked_finger_reports_stall(self):
        """물체에 막힌 손가락은 STALL로 끝나고 나머지는 정상 도달한다."""

        clock, calibration, link, client = make_mock_client(blocked=[1])
        client.connect()
        client.move_fractions([1.0, 1.0, None, None])
        results = client.wait_settled({0, 1}, timeout=10.0)
        self.assertEqual(results[0].kind, "DONE")
        self.assertTrue(results[0].reached)
        self.assertEqual(results[1].kind, "STALL")
        self.assertTrue(results[1].blocked)

    def test_move_outside_soft_limit_is_refused_before_sending(self):
        clock, calibration, link, client = make_mock_client()
        client.connect()
        with self.assertRaises(HandError):
            client.move_adc(0, 4000)          # 소프트 한계 3330 초과
        with self.assertRaises(CalibrationError):
            client.move_fraction(0, 1.5)      # 비율 범위 밖
        with self.assertRaises(CalibrationError):
            client.move_preset("없는자세")     # 모르는 프리셋

    def test_partial_move_leaves_others_untouched(self):
        clock, calibration, link, client = make_mock_client(initial=[3300, 3350, 3400, 450])
        client.connect()
        before = client.positions()
        client.move_fractions([0.5, None, None, None])
        client.wait_settled({0}, timeout=10.0)
        after = client.positions()
        self.assertNotEqual(before[0], after[0])
        for index in (1, 2, 3):
            with self.subTest(finger=index):
                self.assertEqual(before[index], after[index])

    def test_shutdown_sends_stop(self):
        clock, calibration, link, client = make_mock_client()
        sent: list[str] = []
        original = link.write_line

        def record(text: str) -> None:
            sent.append(text)
            original(text)

        link.write_line = record  # type: ignore[assignment]
        with client:
            client.open()
        self.assertEqual(sent[-1], "STOP")

    def test_status_reports_moving_target(self):
        clock, calibration, link, client = make_mock_client(initial=[3300, 3350, 3400, 450])
        client.connect()
        client.move_fraction(0, 1.0)
        entries = client.status()
        self.assertEqual(entries[0].state, "MOVING")
        self.assertEqual(entries[0].target, calibration.finger(0).closed_adc)
        client.stop()


if __name__ == "__main__":
    unittest.main()


class TestPresetSequencing(unittest.TestCase):
    """파지 순서(엄지 받침 먼저)가 지켜지는지 검증한다.

    2026-08-24 실측: 네 손가락을 동시에 닫으면 물체가 밀려나 전부 100 %까지 닫혔고,
    엄지를 먼저 세우면 세 손가락이 78~82 %에서 STALL 되어 실제로 쥐었다.
    """

    def setUp(self):
        self.calibration = load_calibration(CONFIG_PATH)

    def test_sequence_partitions_all_fingers(self):
        for name in ("OPEN", "PRECISION", "WRAP", "POWER"):
            with self.subTest(preset=name):
                steps = self.calibration.preset_steps(name)
                flat = [index for step in steps for index in step]
                self.assertCountEqual(flat, [0, 1, 2, 3])

    def test_thumb_moves_first_for_opposition_poses(self):
        for name in ("POWER", "PRECISION"):
            with self.subTest(preset=name):
                steps = self.calibration.preset_steps(name)
                self.assertGreaterEqual(len(steps), 2)
                self.assertIn(3, steps[0])              # 엄지가 첫 단계
                self.assertNotIn(3, steps[-1])
                self.assertIn(0, steps[-1])             # 검지는 마지막 단계

    def test_missing_finger_in_sequence_is_rejected(self):
        import tempfile

        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "  POWER:     [[3], [0, 1, 2]]", "  POWER:     [[3], [0, 1]]"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_sequence.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(CalibrationError):
                load_calibration(path)

    def test_duplicate_finger_in_sequence_is_rejected(self):
        import tempfile

        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "  POWER:     [[3], [0, 1, 2]]", "  POWER:     [[3], [0, 1, 2, 3]]"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup_sequence.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(CalibrationError):
                load_calibration(path)

    def test_step_moves_only_its_group(self):
        """1단계에서는 엄지만 움직이고 나머지는 그대로 있어야 한다."""

        clock, calibration, link, client = make_mock_client(initial=[3330, 3400, 3420, 420])
        client.connect()
        before = client.positions()

        group = client.move_preset_step("POWER", 0)
        self.assertEqual(group, (3,))
        client.wait_settled(set(group), timeout=10.0)
        middle = client.positions()
        self.assertNotEqual(before[3], middle[3])            # 엄지는 움직였다
        for index in (0, 1, 2):
            with self.subTest(finger=index):
                self.assertEqual(before[index], middle[index])   # 나머지는 그대로

        group = client.move_preset_step("POWER", 1)
        self.assertEqual(group, (0, 1, 2))
        client.wait_settled(set(group), timeout=10.0)
        after = client.positions()
        for index in (0, 1, 2):
            with self.subTest(finger=index, phase="2단계"):
                self.assertNotEqual(middle[index], after[index])

    def test_run_preset_completes_all_steps(self):
        clock, calibration, link, client = make_mock_client(initial=[3330, 3400, 3420, 420])
        client.connect()
        results = client.run_preset("POWER", timeout=10.0)
        self.assertEqual(sorted(results), [0, 1, 2, 3])
        positions = client.positions()
        for finger, fraction in zip(calibration.fingers, calibration.preset("POWER")):
            expected = finger.fraction_to_adc(fraction)
            self.assertLessEqual(
                abs(positions[finger.index] - expected), calibration.deadband_adc
            )
