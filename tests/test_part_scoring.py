"""부위 수준 파지 결정(part_scoring, 파지점 없는 변형) 단위 테스트.

기존 scoring 테스트와 같은 모형으로, 점 기반 결정과 같은 판정(자세·타이브레이크)이
나오는지와 부위 수준 기능부 얽힘 검사가 의도대로 동작하는지 검증한다.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.grasp_selection.candidates import GraspCandidate  # noqa: E402
from src.grasp_selection.part_scoring import decide_grasp_by_part  # noqa: E402
from src.grasp_selection.scoring import HandState, load_selection_config  # noqa: E402

CONFIG = load_selection_config(
    REPOSITORY_ROOT / "configs" / "grasp_selection.yaml",
    REPOSITORY_ROOT / "configs" / "human_grasp_prior.yaml",
)
SHAPE = (480, 640)
NAMES = {0: "handle_grasp_region", 1: "body_grasp_region"}


def circle(center, radius, class_id=0, confidence=0.9) -> GraspCandidate:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return GraspCandidate(class_id=class_id, class_name=NAMES[class_id],
                          confidence=confidence, mask=mask.astype(bool), grasp_point=center,
                          width_px=float(2 * radius), area_px=int(mask.sum()),
                          touches_boundary=False)


def bar(x0, x1, y0, y1, class_id=0, confidence=0.9) -> GraspCandidate:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return GraspCandidate(class_id=class_id, class_name=NAMES[class_id],
                          confidence=confidence, mask=mask, grasp_point=center,
                          width_px=float(y1 - y0), area_px=int(mask.sum()),
                          touches_boundary=False)


NO_FUNCTIONAL = np.zeros(SHAPE, dtype=bool)


class TestPartDecision(unittest.TestCase):
    @staticmethod
    def _mug():
        return [circle((240, 240), 45, class_id=0, confidence=0.55),
                circle((360, 240), 90, class_id=1, confidence=0.9)]

    def test_몸통_쪽_접근은_POWER(self) -> None:
        decision = decide_grasp_by_part(self._mug(), NO_FUNCTIONAL,
                                        HandState(position=(560.0, 240.0)), CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertEqual(decision.pose, "POWER")
        self.assertEqual(decision.candidate.class_id, 1)

    def test_손잡이_쪽_접근은_WRAP(self) -> None:
        decision = decide_grasp_by_part(self._mug(), NO_FUNCTIONAL,
                                        HandState(position=(150.0, 240.0)), CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertEqual(decision.pose, "WRAP")
        self.assertEqual(decision.candidate.class_id, 0)

    def test_후보가_없으면_NO_TARGET(self) -> None:
        decision = decide_grasp_by_part([], NO_FUNCTIONAL,
                                        HandState(position=(100.0, 100.0)), CONFIG)
        self.assertEqual(decision.state, "NO_TARGET")


class TestFunctionalEntanglement(unittest.TestCase):
    """부위 얽힘 검사: 기능부와 '맞닿은' 부위는 통과, '얽힌' 부위만 거부."""

    def test_기능부와_맞닿은_자루는_통과한다(self) -> None:
        handle = bar(200, 320, 236, 246)                     # 도구 자루 (얇은 막대)
        functional = np.zeros(SHAPE, dtype=bool)
        functional[262:274, 200:320] = True                  # 16px 떨어진 날 (얽힘 0%)
        decision = decide_grasp_by_part([handle], functional,
                                        HandState(position=(150.0, 240.0)), CONFIG)
        self.assertEqual(len(decision.ranked), 1)

    def test_기능부를_따라_붙은_조각은_거부된다(self) -> None:
        sliver = bar(200, 320, 236, 246)
        functional = np.zeros(SHAPE, dtype=bool)
        functional[250:262, 200:320] = True                  # 4px 간격 → 픽셀 70%가 12px 이내
        decision = decide_grasp_by_part([sliver], functional,
                                        HandState(position=(150.0, 240.0)), CONFIG)
        self.assertEqual(len(decision.ranked), 0)
        self.assertEqual(sliver.scores.get("reject_reason"), "functional_entangled")

    def test_실측_후보가_있으면_유령이_양보한다(self) -> None:
        """유령 점수 할인: 동등한 조건이면 실측 후보가 1위가 된다 (전환 반응성)."""

        live = circle((240, 240), 40, class_id=1, confidence=0.8)
        ghost = circle((400, 240), 40, class_id=1, confidence=0.8)
        ghost._ghost = True   # type: ignore[attr-defined]
        hand = HandState(position=(320.0, 240.0))   # 두 후보와 등거리
        decision = decide_grasp_by_part([ghost, live], NO_FUNCTIONAL, hand, CONFIG)
        self.assertIs(decision.ranked[0], live)

    def test_유령_혼자면_그대로_선택된다(self) -> None:
        ghost = circle((300, 240), 40, class_id=1, confidence=0.8)
        ghost._ghost = True   # type: ignore[attr-defined]
        decision = decide_grasp_by_part([ghost], NO_FUNCTIONAL,
                                        HandState(position=(500.0, 240.0)), CONFIG)
        self.assertEqual(decision.state, "GRASP")

    def test_유령_후보는_얽힘_거부를_면제받는다(self) -> None:
        ghost = bar(200, 320, 236, 246)
        ghost._ghost = True   # type: ignore[attr-defined]
        functional = np.zeros(SHAPE, dtype=bool)
        functional[250:262, 200:320] = True
        decision = decide_grasp_by_part([ghost], functional,
                                        HandState(position=(150.0, 240.0)), CONFIG)
        self.assertEqual(len(decision.ranked), 1)


if __name__ == "__main__":
    unittest.main()
