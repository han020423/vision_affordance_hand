"""로봇손 추적의 모델 비의존 로직(방향 EMA·스케일·상태 무효화) 단위 테스트."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.perception.hand_tracker import (  # noqa: E402
    HandMotionEstimator,
    HandTracker,
    HandTrackerConfig,
    SegmentationHandSelector,
    load_hand_tracker_config,
)


class TestHandMotionEstimator(unittest.TestCase):
    def test_정지_상태에서는_방향이_없다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=None)
        state = est.observe((100.0, 100.0), 80.0)
        self.assertIsNone(state.direction)
        state = est.observe((100.5, 100.0), 80.0)
        self.assertIsNone(state.direction)

    def test_오른쪽으로_이동하면_방향이_오른쪽이다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.5, min_speed_px=4.0, hand_width_mm=None)
        for x in (100.0, 130.0, 160.0, 190.0):
            state = est.observe((x, 200.0), 80.0)
        self.assertIsNotNone(state.direction)
        self.assertGreater(state.direction[0], 0.95)
        self.assertAlmostEqual(state.direction[1], 0.0, places=3)

    def test_손_폭_실측이_있으면_스케일을_추정한다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=90.0)
        state = est.observe((100.0, 100.0), 180.0)
        self.assertAlmostEqual(state.mm_per_px, 0.5)

    def test_실측이_없으면_스케일은_None이다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=None)
        self.assertIsNone(est.observe((0.0, 0.0), 100.0).mm_per_px)

    def test_잘린_박스가_스케일을_부풀리지_않는다(self) -> None:
        """부분 검출·가장자리 잘림으로 박스가 작아져도 mm/px는 감쇠 최댓값 기준이다
        (2026-08-25 O 실물 시험: 순간값 사용 시 폭 필터가 몸통을 반복 거부 → POWER 불발)."""

        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=120.0,
                                  width_px_decay=0.995)
        state = est.observe((100.0, 100.0), 100.0)
        self.assertAlmostEqual(state.mm_per_px, 1.2)
        # 다음 프레임: 손이 가장자리에 잘려 박스 짧은 변이 절반이 됨
        state = est.observe((105.0, 100.0), 50.0)
        self.assertAlmostEqual(state.mm_per_px, 120.0 / (100.0 * 0.995), places=4)

    def test_손이_커지면_스케일이_즉시_따라간다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=120.0)
        est.observe((100.0, 100.0), 100.0)
        state = est.observe((100.0, 100.0), 150.0)   # 카메라 쪽으로 접근
        self.assertAlmostEqual(state.mm_per_px, 0.8)

    def test_reset이_스케일_기준도_지운다(self) -> None:
        est = HandMotionEstimator(ema_alpha=0.3, min_speed_px=4.0, hand_width_mm=120.0)
        est.observe((100.0, 100.0), 100.0)
        est.reset()
        state = est.observe((100.0, 100.0), 50.0)
        self.assertAlmostEqual(state.mm_per_px, 2.4)   # 새 기준으로 시작


class TestSegmentationHandSelector(unittest.TestCase):
    """검은 정지 물체의 robot_hand 오검출이 손 추적을 납치하지 않는지 검증한다."""

    def test_정지한_손도_기본_설정에서는_즉시_획득된다(self) -> None:
        """움직임 게이트는 정지한 손을 못 잡아 기본 비활성 (2026-08-25 사용자 결정)."""

        sel = SegmentationHandSelector()
        self.assertEqual(sel.select([(320.0, 240.0)], [0.9]), 0)

    def test_획득은_최고_신뢰도_검출을_고른다(self) -> None:
        sel = SegmentationHandSelector()
        self.assertEqual(sel.select([(100.0, 100.0), (500.0, 300.0)], [0.55, 0.95]), 1)

    def test_추적_중_훨씬_강한_검출이_나타나면_갈아탄다(self) -> None:
        """오검출을 잡고 있다가 진짜 손(신뢰도 확실히 높음)이 들어오면 자기 회복한다."""

        sel = SegmentationHandSelector(switch_conf_margin=0.2)
        sel.select([(100.0, 100.0)], [0.6])                 # 오검출 획득
        pick = sel.select([(100.0, 100.0), (600.0, 300.0)], [0.6, 0.95])
        self.assertEqual(pick, 1)

    def test_비슷한_신뢰도면_가까운_쪽을_유지한다(self) -> None:
        sel = SegmentationHandSelector(switch_conf_margin=0.2)
        sel.select([(100.0, 100.0)], [0.85])
        pick = sel.select([(100.0, 100.0), (600.0, 300.0)], [0.85, 0.9])
        self.assertEqual(pick, 0)

    def test_움직임_게이트를_켜면_정지_오검출은_획득되지_않는다(self) -> None:
        sel = SegmentationHandSelector(min_motion_frames=2)
        for _ in range(10):
            self.assertIsNone(sel.select([(320.0, 240.0)], [0.9]))

    def test_움직이는_손은_연속_이동_후_획득된다(self) -> None:
        sel = SegmentationHandSelector(min_motion_px=8.0, min_motion_frames=2)
        picks = [sel.select([(600.0 - 30.0 * k, 300.0)], [0.9]) for k in range(4)]
        self.assertIsNone(picks[0])          # 이력 없음
        self.assertIsNone(picks[1])          # 이동 1프레임
        self.assertEqual(picks[2], 0)        # 이동 2프레임 연속 → 획득
        self.assertEqual(picks[3], 0)        # 이후 유지

    def test_정지_오검출과_함께_있어도_움직이는_쪽을_고른다(self) -> None:
        sel = SegmentationHandSelector(min_motion_px=8.0, min_motion_frames=2)
        static = (100.0, 100.0)
        pick = None
        for k in range(4):
            moving = (600.0 - 30.0 * k, 300.0)
            pick = sel.select([static, moving], [0.95, 0.6])   # 오검출 신뢰도가 더 높아도
        self.assertEqual(pick, 1)

    def test_추적_중_먼_오검출로_점프하지_않는다(self) -> None:
        sel = SegmentationHandSelector(maintain_radius_px=200.0, min_motion_frames=2)
        for k in range(3):
            sel.select([(600.0 - 30.0 * k, 300.0)], [0.9])     # 획득 완료 (540,300)
        # 손 검출이 끊기고 먼 곳의 정지 오검출만 남음 → 채택하지 않는다
        self.assertIsNone(sel.select([(100.0, 100.0)], [0.95]))

    def test_추적_중_잠시_멈춘_손은_유지된다(self) -> None:
        sel = SegmentationHandSelector(min_motion_frames=2)
        for k in range(3):
            sel.select([(600.0 - 30.0 * k, 300.0)], [0.9])
        for _ in range(5):                                     # 같은 자리에 정지
            self.assertEqual(sel.select([(540.0, 300.0)], [0.9]), 0)

    def test_TTL을_넘기면_다시_획득부터_시작한다(self) -> None:
        sel = SegmentationHandSelector(min_motion_frames=2, miss_ttl_frames=3)
        for k in range(3):
            sel.select([(600.0 - 30.0 * k, 300.0)], [0.9])
        for _ in range(4):                                     # 미검출로 TTL 초과
            sel.select([], [])
        # 추적이 버려졌으므로 정지 검출은 다시 거부된다
        self.assertIsNone(sel.select([(540.0, 300.0)], [0.9]))


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        import torch

        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf) if xyxy else None


class _FakeModel:
    """검출 결과를 미리 정해 주는 가짜 검출기."""

    def __init__(self, detections):
        self.detections = list(detections)

    def predict(self, frame, **kwargs):
        item = self.detections.pop(0) if self.detections else None
        if item is None:
            return [_FakeResult([], [])]
        return [_FakeResult([item[0]], [item[1]])]


class TestHandTracker(unittest.TestCase):
    def test_검출_간격과_상태_유지(self) -> None:
        config = HandTrackerConfig(detect_every=3, max_missing_frames=6, min_box_px=10)
        model = _FakeModel([([100, 100, 200, 180], 0.9), ([140, 100, 240, 180], 0.9)])
        tracker = HandTracker(config, REPOSITORY_ROOT, model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = tracker.update(frame)   # 프레임 1: 상태가 없으므로 즉시 검출
        self.assertIsNotNone(first)
        self.assertEqual(first.position, (150.0, 140.0))
        second = tracker.update(frame)  # 프레임 2: 검출 간격 전 → 상태 유지
        self.assertEqual(second.position, first.position)
        third = tracker.update(frame)   # 프레임 3: 두 번째 검출 → 위치 갱신
        self.assertEqual(third.position, (190.0, 140.0))

    def test_오래_놓치면_상태를_무효화한다(self) -> None:
        config = HandTrackerConfig(detect_every=1, max_missing_frames=3, min_box_px=10)
        model = _FakeModel([([100, 100, 200, 180], 0.9), None, None, None])
        tracker = HandTracker(config, REPOSITORY_ROOT, model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIsNotNone(tracker.update(frame))
        tracker.update(frame)
        tracker.update(frame)
        self.assertIsNone(tracker.update(frame))

    def test_너무_작은_박스는_무시한다(self) -> None:
        config = HandTrackerConfig(detect_every=1, min_box_px=40)
        model = _FakeModel([([10, 10, 30, 30], 0.95)])
        tracker = HandTracker(config, REPOSITORY_ROOT, model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIsNone(tracker.update(frame))


class TestConfig(unittest.TestCase):
    def test_설정_파일이_로드된다(self) -> None:
        config = load_hand_tracker_config(REPOSITORY_ROOT / "configs" / "hardware" / "hand_tracker.yaml")
        self.assertEqual(config.backend, "motion")
        self.assertTrue(config.motion_use_framediff)
        self.assertIn("robot hand", config.prompts)


if __name__ == "__main__":
    unittest.main()
