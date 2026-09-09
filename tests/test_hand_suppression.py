"""로봇손 오검출 억제 필터 시험 (src/grasp_selection/hand_filter.py).

2026-08-24 젯슨 실물 시험에서 확인된 문제(로봇손이 파지 후보로 오인됨)에 대한 방어를
검증한다: 정지 후보만 통과, 움직이는 후보 차단, 손 박스 겹침 거부, 접근 중 목표 보호.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.grasp_selection import HandSuppressionFilter  # noqa: E402
from src.grasp_selection.candidates import GraspCandidate  # noqa: E402

SHAPE = (480, 640)


def circle_candidate(center, radius=30, class_id=0, confidence=0.9) -> GraspCandidate:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    names = {0: "handle_grasp_region", 1: "body_grasp_region"}
    return GraspCandidate(
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        mask=mask.astype(bool),
        grasp_point=center,
        width_px=float(radius * 2),
        area_px=int(mask.sum()),
        touches_boundary=False,
    )


class TestStability(unittest.TestCase):
    def test_정지_후보는_지정_프레임_후에만_통과한다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=3)
        static = (300, 240)
        for frame in range(1, 6):
            kept, suppressed = f.update([circle_candidate(static)])
            with self.subTest(frame=frame):
                if frame < 3:
                    self.assertEqual(len(kept), 0)
                    self.assertEqual(suppressed["unstable"], 1)
                else:
                    self.assertEqual(len(kept), 1)

    def test_움직이는_후보는_계속_차단된다(self) -> None:
        """화면을 지나가는 손: 매 프레임 80px 이동 → 트랙이 이어지지 않아야 한다."""

        f = HandSuppressionFilter(min_stable_frames=3, match_radius_px=60)
        for frame in range(8):
            kept, suppressed = f.update([circle_candidate((100 + frame * 80, 240))])
            with self.subTest(frame=frame):
                self.assertEqual(len(kept), 0)
                self.assertEqual(suppressed["unstable"], 1)

    def test_약간의_흔들림은_같은_후보로_본다(self) -> None:
        """마스크 잡음으로 파지점이 몇 px 흔들려도 정지 물체로 인정한다."""

        f = HandSuppressionFilter(min_stable_frames=3, match_radius_px=60)
        jitter = [(300, 240), (305, 243), (298, 238), (303, 241)]
        kept = []
        for point in jitter:
            kept, _ = f.update([circle_candidate(point)])
        self.assertEqual(len(kept), 1)

    def test_검출_깜빡임은_TTL_안에서_용서된다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=3, miss_ttl_frames=10)
        static = (300, 240)
        f.update([circle_candidate(static)])
        f.update([])                       # 한 프레임 검출 실패
        f.update([circle_candidate(static)])
        kept, _ = f.update([circle_candidate(static)])
        self.assertEqual(len(kept), 1)     # 나이 3 도달

    def test_클래스가_뒤집혀도_트랙이_유지되고_다수결로_고정된다(self) -> None:
        """종이컵 대응: 같은 자리 후보는 클래스가 바뀌어도 같은 트랙이고,
        후보 클래스는 트랙 다수결(선점 클래스)로 덮어쓴다."""

        f = HandSuppressionFilter(min_stable_frames=2)
        f.update([circle_candidate((300, 240), class_id=0)])
        kept, _ = f.update([circle_candidate((300, 240), class_id=1)])
        self.assertEqual(len(kept), 1)                     # 트랙 유지 → 나이 2
        self.assertEqual(kept[0].class_id, 0)              # 다수결 클래스로 고정
        self.assertEqual(kept[0].class_name, "handle_grasp_region")

    def test_교대로_뒤집히는_클래스는_처음_클래스로_안정된다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1, class_vote_window=7, class_switch_min=5)
        kept = []
        for frame in range(10):
            kept, _ = f.update([circle_candidate((300, 240), class_id=frame % 2)])
        self.assertEqual(kept[0].class_id, 0)              # 교대 관측으로는 전환 없음

    def test_지속적인_클래스_변경은_결국_전환된다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1, class_vote_window=7, class_switch_min=5)
        f.update([circle_candidate((300, 240), class_id=0)])
        kept = []
        for _ in range(5):                                 # 5프레임 연속 반대 클래스
            kept, _ = f.update([circle_candidate((300, 240), class_id=1)])
        self.assertEqual(kept[0].class_id, 1)

    def test_인접한_손잡이와_몸통은_서로_다른_트랙을_유지한다(self) -> None:
        """클래스 무관 매칭이 같은 프레임의 인접 부위를 한 트랙으로 합치면 안 된다."""

        f = HandSuppressionFilter(min_stable_frames=2, match_radius_px=60)
        kept = []
        for _ in range(3):
            kept, _ = f.update([
                circle_candidate((240, 240), radius=30, class_id=0),   # 파지점 간 거리 50px
                circle_candidate((290, 240), radius=30, class_id=1),
            ])
        self.assertEqual(len(kept), 2)
        self.assertEqual({c.class_id for c in kept}, {0, 1})           # 클래스 보존


def make_candidate(mask: np.ndarray, grasp_point, class_id=0, confidence=0.9) -> GraspCandidate:
    """임의 마스크·파지점으로 후보를 만든다 (가림 시나리오 구성용)."""

    names = {0: "handle_grasp_region", 1: "body_grasp_region"}
    return GraspCandidate(
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        mask=mask.astype(bool),
        grasp_point=grasp_point,
        width_px=30.0,
        area_px=int(mask.sum()),
        touches_boundary=False,
    )


def occluded_circle(center, radius, cut_x, grasp_point) -> GraspCandidate:
    """오른쪽(x > cut_x)이 손에 가려진 원형 후보: 면적 급감 + 파지점 이동."""

    mask = np.zeros(SHAPE, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    mask[:, cut_x + 1:] = 0
    return make_candidate(mask, grasp_point)


class TestOcclusionRobustness(unittest.TestCase):
    """2026-08-25 O 실물 시험에서 확인된 문제: 손이 물체를 가리면 마스크가 잘려
    파지점이 60px 이상 튀고 면적이 급감 → 트랙이 끊겨 unstable로 거부(점 소실)."""

    def test_가림으로_잘린_마스크도_트랙이_이어진다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=3)
        for _ in range(3):
            kept, _ = f.update([circle_candidate((300, 240), radius=60)])
        self.assertEqual(len(kept), 1)
        # 손이 오른쪽을 가림: 파지점 70px 이동(>60) — 옛 파지점(300,240)은 아직 마스크 안
        kept, suppressed = f.update([occluded_circle((300, 240), 60, cut_x=310,
                                                     grasp_point=(230, 240))])
        self.assertEqual(len(kept), 1, "가림으로 트랙이 끊기면 안 된다")
        self.assertEqual(suppressed["unstable"], 0)

    def test_옛_파지점을_포함해도_과도_성장은_차단한다(self) -> None:
        """오검출 병합(거대 마스크)은 옛 점을 포함해도 같은 트랙으로 보지 않는다."""

        f = HandSuppressionFilter(min_stable_frames=3, area_ratio_max=2.5)
        for _ in range(3):
            f.update([circle_candidate((300, 240), radius=30)])
        giant = np.ones(SHAPE, dtype=np.uint8)     # 면적이 full_area의 2.5배 초과
        kept, suppressed = f.update([make_candidate(giant, (300, 240))])
        self.assertEqual(len(kept), 0)
        self.assertEqual(suppressed["unstable"], 1)

    def test_가림_중에는_파지점을_유지한다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1, occlusion_hold_ratio=0.7)
        kept, _ = f.update([circle_candidate((300, 240), radius=50)])
        f.stabilize_points(kept)
        self.assertEqual(kept[0].grasp_point, (300, 240))
        # 다음 프레임: 절반 가림 → 새 점이 가림 경계 쪽으로 튀지만 유지돼야 한다
        kept, _ = f.update([occluded_circle((300, 240), 50, cut_x=300,
                                            grasp_point=(270, 240))])
        held = f.stabilize_points(kept)
        self.assertEqual(held, 1)
        self.assertEqual(kept[0].grasp_point, (300, 240), "가림 전 파지점을 유지해야 한다")
        self.assertEqual(kept[0].scores.get("point_held"), 1.0)

    def test_비가림_상태에서는_EMA로_떨림을_줄인다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1, point_ema_alpha=0.4)
        kept, _ = f.update([circle_candidate((300, 240), radius=50)])
        f.stabilize_points(kept)
        kept, _ = f.update([circle_candidate((306, 242), radius=50)])
        f.stabilize_points(kept)
        # 0.6·(300,240) + 0.4·(306,242) = (302.4, 240.8) → 반올림 (302, 241)
        self.assertEqual(kept[0].grasp_point, (302, 241))
        self.assertIsNone(kept[0].scores.get("point_held"))


class TestGhostCandidate(unittest.TestCase):
    """유령 후보 (2026-08-26): 최종 접근에서 후보가 통째로 사라져도 마지막 검증
    스냅샷으로 파지점을 유지한다. 스냅샷은 stabilize_points(=순위 도달)에서 저장된다."""

    def _stable_filter(self, **kwargs):
        f = HandSuppressionFilter(min_stable_frames=2, **kwargs)
        kept = []
        for _ in range(2):
            kept, _ = f.update([circle_candidate((300, 240), radius=40, class_id=1)])
        f.stabilize_points(kept)           # 검증 통과 → 스냅샷 저장
        return f

    def test_후보가_사라지면_유령이_대신한다(self) -> None:
        f = self._stable_filter()
        kept, suppressed = f.update([])    # 모델 미검출 (근접 가림)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].class_id, 1)
        self.assertEqual(suppressed["ghost"], 1)
        f.stabilize_points(kept)
        self.assertEqual(kept[0].scores.get("point_held"), 1.0)   # HOLD 표시

    def test_손_박스_거부에도_유령이_나온다(self) -> None:
        """최종 접근: 목표가 손 박스 안에 들어가 거부돼도 파지점은 유지돼야 한다."""

        f = self._stable_filter()
        hand_box = (240, 180, 360, 300)    # 목표를 완전히 덮는 손 박스
        kept, suppressed = f.update([circle_candidate((300, 240), radius=40, class_id=1)],
                                    hand_box=hand_box)
        self.assertEqual(suppressed["hand_overlap"], 1)
        self.assertEqual(len(kept), 1)     # 실측 후보는 거부됐지만 유령이 대신
        self.assertEqual(suppressed["ghost"], 1)

    def test_유령은_TTL까지만_유지된다(self) -> None:
        f = self._stable_filter(ghost_ttl_frames=3)
        counts = []
        for _ in range(5):
            kept, _ = f.update([])
            counts.append(len(kept))
        self.assertEqual(counts, [1, 1, 1, 0, 0])

    def test_손이_덮고_있는_동안은_유지가_끝나지_않는다(self) -> None:
        """호버·정렬이 아무리 길어져도, 손이 목표를 덮고 있는 한 파지점은 유지된다."""

        f = self._stable_filter(ghost_ttl_frames=3)
        hand_box = (240, 180, 360, 300)   # 트랙(300,240)을 덮는 손 박스
        for _ in range(10):               # TTL(3)보다 훨씬 길게 덮고 있어도
            _, suppressed = f.update([], hand_box=hand_box)
            self.assertEqual(suppressed["ghost"], 1)
        counts = [f.update([])[1]["ghost"] for _ in range(5)]   # 손이 떠난 뒤부터 TTL
        self.assertEqual(counts, [1, 1, 1, 0, 0])

    def test_검증_전_트랙은_유령이_없다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=2)
        f.update([circle_candidate((300, 240), radius=40)])   # 나이 1 (unstable)
        kept, suppressed = f.update([])
        self.assertEqual(len(kept), 0)
        self.assertEqual(suppressed["ghost"], 0)

    def test_다른_물체로_교체되면_유령이_즉시_사라진다(self) -> None:
        """옛 자리를 '다른' 안정 후보가 덮으면 교체로 보고 기억을 폐기한다."""

        f = self._stable_filter()   # (300,240) r40 body 트랙 + 스냅샷
        # 같은 자리에 훨씬 큰 새 물체 (면적 2.5배 초과 → 기존 트랙과 매칭 불가)
        replacement = circle_candidate((300, 240), radius=70, class_id=0)
        f.update([replacement])                    # 새 트랙 나이 1 (아직 불안정)
        kept, suppressed = f.update([circle_candidate((300, 240), radius=70, class_id=0)])
        self.assertEqual(len(kept), 1)             # 새 물체만 (나이 2 도달)
        self.assertEqual(suppressed["ghost"], 0)   # 옛 유령은 교체 증거로 폐기됨
        kept, suppressed = f.update([])            # 새 물체도 사라져도
        self.assertEqual(suppressed["ghost"], 0)   # 옛 기억은 되살아나지 않는다

    def test_실측_후보가_돌아오면_유령이_물러난다(self) -> None:
        f = self._stable_filter()
        f.update([])                                          # 유령 1프레임
        kept, suppressed = f.update([circle_candidate((300, 240), radius=40, class_id=1)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(suppressed["ghost"], 0)              # 실측 복귀 → 유령 없음
        self.assertFalse(getattr(kept[0], "_ghost", False))


class TestHandOverlap(unittest.TestCase):
    def test_손_박스_안_후보는_즉시_거부된다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1, hand_overlap_max=0.5)
        candidate = circle_candidate((300, 240), radius=30)
        hand_box = (240, 180, 360, 300)    # 후보를 완전히 덮는 박스
        for _ in range(5):                 # 오래 있어도 계속 거부
            kept, suppressed = f.update([candidate], hand_box=hand_box)
        self.assertEqual(len(kept), 0)
        self.assertEqual(suppressed["hand_overlap"], 1)

    def test_살짝_겹치는_목표는_지킨다(self) -> None:
        """접근 중 손 박스가 목표물 가장자리에 닿는 정도로는 거부하지 않는다."""

        f = HandSuppressionFilter(min_stable_frames=1, hand_overlap_max=0.5, hand_box_margin=0.0)
        candidate = circle_candidate((300, 240), radius=30)
        hand_box = (150, 180, 285, 300)    # 후보 왼쪽 일부만 덮음 (< 50%)
        kept, suppressed = f.update([candidate], hand_box=hand_box)
        self.assertEqual(len(kept), 1)
        self.assertEqual(suppressed["hand_overlap"], 0)

    def test_박스가_없으면_안정성만_적용된다(self) -> None:
        f = HandSuppressionFilter(min_stable_frames=1)
        kept, _ = f.update([circle_candidate((300, 240))], hand_box=None)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
