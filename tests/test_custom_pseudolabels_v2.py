"""가위·드라이버용 앵커-잔여 반자동 후보 조립의 단위 테스트."""

from __future__ import annotations

import unittest

import numpy as np

from src.labeling.custom_pseudolabels import (
    combine_category_candidates,
    split_anchor_residual,
    split_scissors_by_rings,
    split_screwdriver_by_width,
)


def _rect(height: int, width: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class SplitAnchorResidualTest(unittest.TestCase):
    def test_가위형_앵커_두_개와_잔여를_분리한다(self) -> None:
        whole = _rect(100, 200, 20, 80, 20, 180)
        # 고리 두 개(왼쪽)와 날(오른쪽)에 해당하는 앵커 두 블록
        anchor = _rect(100, 200, 25, 45, 25, 55) | _rect(100, 200, 55, 75, 25, 55)
        components, residual, flags, diagnostics = split_anchor_residual(
            whole, anchor, max_anchor_instances=2, minimum_pixels=64
        )
        self.assertEqual(len(components), 2)
        self.assertEqual(diagnostics["anchor_instances"], 2)
        self.assertTrue(residual.any())
        # 잔여는 앵커와 겹치지 않고 전체 안에 있어야 한다
        union = np.zeros_like(whole)
        for component in components:
            self.assertFalse((component & residual).any())
            union |= component
        self.assertTrue(((union | residual) <= whole).all())
        self.assertEqual(flags, [])

    def test_앵커가_없으면_검수_표시를_남긴다(self) -> None:
        whole = _rect(50, 50, 10, 40, 10, 40)
        empty = np.zeros_like(whole)
        components, residual, flags, _ = split_anchor_residual(
            whole, empty, max_anchor_instances=1, minimum_pixels=64
        )
        self.assertEqual(components, [])
        self.assertIn("앵커_부위_후보_누락", flags)
        self.assertTrue(residual.any())

    def test_앵커가_전체를_벗어나면_교집합만_사용한다(self) -> None:
        whole = _rect(60, 60, 10, 50, 10, 50)
        anchor = _rect(60, 60, 0, 60, 0, 20)  # 전체 밖 영역 포함
        components, residual, _, _ = split_anchor_residual(
            whole, anchor, max_anchor_instances=1, minimum_pixels=64
        )
        self.assertEqual(len(components), 1)
        self.assertTrue((components[0] <= whole).all())
        self.assertFalse((components[0] & residual).any())


class SplitScissorsByRingsTest(unittest.TestCase):
    def _가위_마스크(self) -> np.ndarray:
        """왼쪽에 고리 두 개(구멍 포함), 오른쪽에 날이 있는 합성 가위."""

        mask = np.zeros((120, 300), dtype=bool)
        # 고리 두 개: 바깥 사각형에서 안쪽 구멍을 뺀 형태
        mask[20:55, 20:70] = True
        mask[28:47, 30:60] = False   # 위 고리 구멍
        mask[65:100, 20:70] = True
        mask[73:92, 30:60] = False   # 아래 고리 구멍
        # 연결부와 날
        mask[50:70, 60:120] = True
        mask[52:68, 120:280] = True  # 날
        return mask

    def test_고리와_날을_분리한다(self) -> None:
        rings, blade, flags, diagnostics = split_scissors_by_rings(
            self._가위_마스크(), minimum_pixels=64
        )
        self.assertEqual(diagnostics["hole_count"], 2)
        self.assertGreaterEqual(len(rings), 1)
        self.assertTrue(blade.any())
        # 날 영역은 오른쪽(열 120 이후)에 있어야 한다
        _, blade_xs = np.where(blade)
        self.assertGreater(float(blade_xs.mean()), 120.0)
        for ring in rings:
            self.assertFalse((ring & blade).any())

    def test_구멍이_없으면_잔여만_남기고_표시한다(self) -> None:
        solid = _rect(60, 200, 20, 40, 10, 190)
        rings, blade, flags, _ = split_scissors_by_rings(solid, minimum_pixels=64)
        self.assertEqual(rings, [])
        self.assertTrue(blade.any())
        self.assertIn("가위_고리_구멍_미검출", flags)


class SplitScrewdriverByWidthTest(unittest.TestCase):
    def test_손잡이와_축을_폭으로_분리한다(self) -> None:
        mask = np.zeros((100, 300), dtype=bool)
        mask[30:70, 20:120] = True    # 두꺼운 손잡이 (폭 40)
        mask[46:54, 120:280] = True   # 가는 축 (폭 8)
        handles, shaft, flags, diagnostics = split_screwdriver_by_width(
            mask, minimum_pixels=64
        )
        self.assertEqual(len(handles), 1)
        self.assertTrue(shaft.any())
        _, handle_xs = np.where(handles[0])
        _, shaft_xs = np.where(shaft)
        self.assertLess(float(handle_xs.mean()), float(shaft_xs.mean()))
        self.assertFalse((handles[0] & shaft).any())
        self.assertEqual(flags, [])

    def test_손잡이가_오른쪽이어도_방향을_맞춘다(self) -> None:
        mask = np.zeros((100, 300), dtype=bool)
        mask[46:54, 20:180] = True    # 가는 축 (왼쪽)
        mask[30:70, 180:280] = True   # 두꺼운 손잡이 (오른쪽)
        handles, shaft, _, _ = split_screwdriver_by_width(mask, minimum_pixels=64)
        self.assertEqual(len(handles), 1)
        _, handle_xs = np.where(handles[0])
        _, shaft_xs = np.where(shaft)
        self.assertGreater(float(handle_xs.mean()), float(shaft_xs.mean()))

    def test_폭_변화가_없으면_표시를_남긴다(self) -> None:
        uniform = _rect(60, 300, 20, 40, 10, 290)
        handles, shaft, flags, _ = split_screwdriver_by_width(uniform, minimum_pixels=64)
        self.assertEqual(handles, [])
        self.assertIn("드라이버_폭_경계_미검출", flags)


class CombineCategoryCandidatesTest(unittest.TestCase):
    def test_잔여를_functional로_조립한다(self) -> None:
        shape = (80, 120)
        anchor = _rect(*shape, 10, 40, 10, 50)
        residual = _rect(*shape, 10, 40, 60, 110)
        semantic, instance, components, flags = combine_category_candidates(
            [anchor],
            residual,
            shape,
            anchor_part="handle",
            residual_part="shaft",
            residual_label="functional_region",
            score=0.9,
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
            minimum_pixels=64,
        )
        self.assertEqual(len(components), 2)
        self.assertEqual(components[0]["label"], "grasp_region")
        self.assertEqual(components[1]["label"], "functional_region")
        self.assertEqual(int(semantic[20, 20]), 1)   # 앵커=grasp 저장값
        self.assertEqual(int(semantic[20, 80]), 2)   # 잔여=functional 저장값
        self.assertNotEqual(int(instance[20, 20]), int(instance[20, 80]))
        self.assertEqual(flags, [])

    def test_고리_두_개는_별도_인스턴스를_유지한다(self) -> None:
        shape = (100, 100)
        ring1 = _rect(*shape, 10, 30, 10, 40)
        ring2 = _rect(*shape, 40, 60, 10, 40)
        blade = _rect(*shape, 10, 60, 50, 90)
        _, instance, components, _ = combine_category_candidates(
            [ring1, ring2],
            blade,
            shape,
            anchor_part="handle",
            residual_part="blade",
            residual_label="functional_region",
            score=0.5,
            box_xyxy=(0.0, 0.0, 1.0, 1.0),
            minimum_pixels=64,
        )
        self.assertEqual([c["part"] for c in components], ["handle", "handle", "blade"])
        self.assertEqual(len({int(c["instance_id"]) for c in components}), 3)
        self.assertEqual(int(instance[20, 20]), 1)
        self.assertEqual(int(instance[50, 20]), 2)

    def test_파지_부위가_없으면_검수_표시를_남긴다(self) -> None:
        shape = (60, 60)
        residual = _rect(*shape, 10, 50, 10, 50)
        _, _, components, flags = combine_category_candidates(
            [],
            residual,
            shape,
            anchor_part="handle",
            residual_part="shaft",
            residual_label="functional_region",
            score=0.3,
            box_xyxy=(0.0, 0.0, 1.0, 1.0),
            minimum_pixels=64,
        )
        self.assertEqual(len(components), 1)
        self.assertIn("파지_부위_없음", flags)

    def test_허용되지_않는_잔여_라벨은_거부한다(self) -> None:
        with self.assertRaises(ValueError):
            combine_category_candidates(
                [],
                np.zeros((10, 10), dtype=bool),
                (10, 10),
                anchor_part="handle",
                residual_part="shaft",
                residual_label="ignore",
                score=0.1,
                box_xyxy=(0.0, 0.0, 1.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
