"""파지 후보 선택 모듈 단위 테스트 (AGENTS 필수 테스트 5·7 대응)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.grasp_selection.candidates import GraspCandidate  # noqa: E402
from src.grasp_selection.scoring import (  # noqa: E402
    Decision,
    HandState,
    decide_grasp,
    load_selection_config,
)

CONFIG = load_selection_config(
    REPOSITORY_ROOT / "configs" / "grasp_selection.yaml",
    REPOSITORY_ROOT / "configs" / "human_grasp_prior.yaml",
)
SHAPE = (480, 640)


def make_bar_candidate(center, length, thickness, class_id=0, confidence=0.9) -> GraspCandidate:
    """가로로 긴 막대 마스크 후보(펜·자루 모형)를 만든다."""

    mask = np.zeros(SHAPE, dtype=np.uint8)
    x, y = center
    cv2.rectangle(mask, (x - length // 2, y - thickness // 2),
                  (x + length // 2, y + thickness // 2), 1, thickness=-1)
    names = {0: "handle_grasp_region", 1: "body_grasp_region"}
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_value, _, max_location = cv2.minMaxLoc(distance)
    return GraspCandidate(
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        mask=mask.astype(bool),
        grasp_point=(int(max_location[0]), int(max_location[1])),
        width_px=float(2.0 * max_value),
        area_px=int(mask.sum()),
        touches_boundary=False,
    )


def make_candidate(center, radius, class_id=0, confidence=0.9) -> GraspCandidate:
    """원형 마스크 후보를 만든다."""

    mask = np.zeros(SHAPE, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    names = {0: "handle_grasp_region", 1: "body_grasp_region"}
    return GraspCandidate(
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        mask=mask.astype(bool),
        grasp_point=center,
        width_px=float(2 * radius),
        area_px=int(mask.sum()),
        touches_boundary=False,
    )


class TestConfig(unittest.TestCase):
    def test_가중치_합은_1이다(self) -> None:
        self.assertAlmostEqual(sum(CONFIG["weights"].values()), 1.0, places=6)

    def test_인간_prior가_로드된다(self) -> None:
        self.assertAlmostEqual(sum(CONFIG["human_prior"].values()), 1.0, places=2)


class TestDecideGrasp(unittest.TestCase):
    def setUp(self) -> None:
        self.no_functional = np.zeros(SHAPE, dtype=bool)

    def test_가까운_후보가_선택된다(self) -> None:
        near = make_candidate((150, 240), 18)
        far = make_candidate((550, 240), 18)
        hand = HandState(position=(100.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([near, far], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertIs(decision.candidate, near)

    def test_점수는_0과_1_사이다(self) -> None:
        candidate = make_candidate((300, 240), 18)
        hand = HandState(position=(100.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([candidate], self.no_functional, hand, CONFIG)
        for key in ("confidence", "distance", "approach", "width_fit",
                    "functional_safety", "human_prior", "total"):
            value = float(candidate.scores[key])
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)
        self.assertEqual(decision.state, "GRASP")

    def test_기능영역_인접_후보는_거부된다(self) -> None:
        candidate = make_candidate((300, 240), 18)
        functional = np.zeros(SHAPE, dtype=bool)
        functional[230:250, 295:305] = True  # 파지점 바로 옆
        hand = HandState(position=(100.0, 240.0))
        decision = decide_grasp([candidate], functional, hand, CONFIG)
        self.assertEqual(decision.state, "NO_TARGET")
        self.assertEqual(candidate.scores.get("reject_reason"), "too_close_to_functional")

    def test_캘리브레이션_시_과대_폭_후보는_거부된다(self) -> None:
        import copy

        calibrated = copy.deepcopy(CONFIG)
        calibrated["pose"]["mm_per_px"] = 0.5  # 폭 300px → 150mm > 최대 110mm
        huge = make_candidate((320, 240), 150)
        hand = HandState(position=(100.0, 240.0))
        decision = decide_grasp([huge], self.no_functional, hand, calibrated)
        self.assertEqual(decision.state, "NO_TARGET")
        self.assertEqual(huge.scores.get("reject_reason"), "max_grasp_width_exceeded")

    def test_캘리브레이션_전에는_폭으로_거부하지_않는다(self) -> None:
        huge = make_candidate((320, 240), 150)
        hand = HandState(position=(100.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([huge], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")

    def test_동점이면_ALIGN이다(self) -> None:
        left = make_candidate((250, 200), 18)
        right = make_candidate((250, 280), 18)  # 손에서 같은 거리·같은 폭
        hand = HandState(position=(250.0, 240.0), direction=None)
        decision = decide_grasp([left, right], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "ALIGN")

    def test_몸통은_POWER_자세다(self) -> None:
        body = make_candidate((300, 240), 60, class_id=1)
        hand = HandState(position=(250.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([body], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertEqual(decision.pose, "POWER")

    # ---- 자세 결정 (VISOR 기하 분석 기반 구조 규칙) ----

    def test_몸통에_붙은_손잡이는_WRAP이다(self) -> None:
        """머그: 손잡이 후보가 몸통 영역과 인접하면 용기 손잡이 → WRAP."""

        handle = make_candidate((200, 240), 25, class_id=0)
        body = make_candidate((280, 240), 60, class_id=1)   # 손잡이 바로 옆 몸통
        hand = HandState(position=(120.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([handle, body], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertIs(decision.candidate, handle)           # 손에 가까운 손잡이가 이김
        self.assertEqual(decision.pose, "WRAP")

    def test_길쭉한_단독_손잡이는_PRECISION이다(self) -> None:
        """펜·드라이버 자루: 몸통 없이 길쭉한 손잡이 → PRECISION."""

        bar = make_bar_candidate((320, 240), 220, 24)       # 종횡비 약 9
        hand = HandState(position=(150.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([bar], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.pose, "PRECISION")

    def test_기능영역이_붙은_손잡이도_문맥으로_PRECISION이다(self) -> None:
        """가위: 둥근 손잡이라도 길쭉한 기능영역(날)이 붙으면 도구 → PRECISION."""

        handle = make_candidate((200, 240), 28, class_id=0)  # 종횡비 1.0 원형
        functional = np.zeros(SHAPE, dtype=np.uint8)
        cv2.rectangle(functional, (230, 228), (560, 252), 1, thickness=-1)  # 날
        hand = HandState(position=(120.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([handle], functional.astype(bool), hand, CONFIG)
        self.assertEqual(decision.pose, "PRECISION")

    def test_단독_둥근_손잡이는_WRAP이다(self) -> None:
        """몸통도 기능영역도 없는 둥근 후보는 기본 WRAP."""

        blob = make_candidate((300, 240), 40)
        hand = HandState(position=(250.0, 240.0), direction=(1.0, 0.0))
        decision = decide_grasp([blob], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.pose, "WRAP")

    def test_자세는_거리와_무관하다(self) -> None:
        """같은 모양을 절반 크기로 줄여도(먼 카메라) 자세가 바뀌지 않는다."""

        for scale in (1.0, 0.5):
            with self.subTest(scale=scale):
                bar = make_bar_candidate((320, 240), int(220 * scale), max(8, int(24 * scale)))
                hand = HandState(position=(150.0, 240.0), direction=(1.0, 0.0))
                decision = decide_grasp([bar], self.no_functional, hand, CONFIG)
                self.assertEqual(decision.pose, "PRECISION")

    def test_후보가_없으면_NO_TARGET이다(self) -> None:
        hand = HandState(position=(100.0, 100.0))
        decision = decide_grasp([], self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "NO_TARGET")


class TestAdjacentTiebreak(unittest.TestCase):
    """인접 부위 타이브레이크 (2026-08-25 파란 머그 POWER/WRAP 요동 대응).

    같은 물체의 손잡이-몸통 사이에서는 점수 대신 손이 향하는 쪽이 승자다."""

    def setUp(self) -> None:
        self.no_functional = np.zeros(SHAPE, dtype=bool)

    @staticmethod
    def _mug() -> list[GraspCandidate]:
        """머그 모형: 몸통(신뢰도 높음)과 그 왼쪽에 붙은 손잡이(신뢰도 낮음)."""

        body = make_candidate((360, 240), 90, class_id=1, confidence=0.9)
        handle = make_candidate((240, 240), 45, class_id=0, confidence=0.55)
        return [handle, body]

    def test_손이_몸통_쪽이면_POWER다(self) -> None:
        hand = HandState(position=(560.0, 240.0))   # 오른쪽(몸통 쪽)에서 접근
        decision = decide_grasp(self._mug(), self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertEqual(decision.candidate.class_id, 1)
        self.assertEqual(decision.pose, "POWER")
        self.assertEqual(decision.candidate.scores.get("adjacent_tiebreak"), 1.0)

    def test_손이_손잡이_쪽이면_신뢰도가_낮아도_WRAP이다(self) -> None:
        hand = HandState(position=(150.0, 240.0))   # 왼쪽(손잡이 쪽)에서 접근
        decision = decide_grasp(self._mug(), self.no_functional, hand, CONFIG)
        self.assertEqual(decision.state, "GRASP")
        self.assertEqual(decision.candidate.class_id, 0)
        self.assertEqual(decision.pose, "WRAP")

    def test_승자가_1위로_재정렬된다(self) -> None:
        hand = HandState(position=(150.0, 240.0))
        decision = decide_grasp(self._mug(), self.no_functional, hand, CONFIG)
        self.assertIs(decision.ranked[0], decision.candidate)

    def test_떨어진_물체_사이에는_적용되지_않는다(self) -> None:
        """인접하지 않은 서로 다른 물체는 기존 점수식으로 결정한다."""

        handle = make_candidate((120, 240), 40, class_id=0, confidence=0.9)
        body = make_candidate((520, 240), 60, class_id=1, confidence=0.9)
        hand = HandState(position=(100.0, 240.0))
        decision = decide_grasp([handle, body], self.no_functional, hand, CONFIG)
        for candidate in decision.ranked:
            self.assertIsNone(candidate.scores.get("adjacent_tiebreak"))

    def test_설정으로_끌_수_있다(self) -> None:
        config = {**CONFIG, "decision": {**CONFIG["decision"], "adjacent_tiebreak": False}}
        hand = HandState(position=(150.0, 240.0))
        decision = decide_grasp(self._mug(), self.no_functional, hand, config)
        for candidate in decision.ranked:
            self.assertIsNone(candidate.scores.get("adjacent_tiebreak"))


class TestGhostExemption(unittest.TestCase):
    """유령 후보(가림 유지 스냅샷)는 스냅샷 시점에 검증을 통과했으므로,
    가림 중 오염될 수 있는 거부 검사(기능부 근접 등)를 면제받는다."""

    @staticmethod
    def _functional_near() -> np.ndarray:
        mask = np.zeros(SHAPE, dtype=np.uint8)
        cv2.circle(mask, (315, 240), 20, 1, thickness=-1)   # 후보 파지점 바로 옆
        return mask.astype(bool)

    def test_일반_후보는_기능부_근접으로_거부된다(self) -> None:
        candidate = make_candidate((300, 240), 30, class_id=1)
        hand = HandState(position=(200.0, 240.0))
        decision = decide_grasp([candidate], self._functional_near(), hand, CONFIG)
        self.assertEqual(len(decision.ranked), 0)
        self.assertEqual(candidate.scores.get("reject_reason"), "too_close_to_functional")

    def test_유령_후보는_거부를_면제받는다(self) -> None:
        ghost = make_candidate((300, 240), 30, class_id=1)
        ghost._ghost = True   # type: ignore[attr-defined]
        hand = HandState(position=(200.0, 240.0))
        decision = decide_grasp([ghost], self._functional_near(), hand, CONFIG)
        self.assertEqual(len(decision.ranked), 1)


if __name__ == "__main__":
    unittest.main()
