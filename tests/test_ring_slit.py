"""고리형(절개선) 폴리곤 변환 단위 테스트."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.yolo_export import component_to_yolo_polygon  # noqa: E402


def make_ring_instance(size: int = 96, outer: int = 36, inner: int = 18) -> np.ndarray:
    """가운데 구멍이 뚫린 고리형 컴포넌트(ID=1) 인스턴스 마스크를 만든다."""

    mask = np.zeros((size, size), dtype=np.uint16)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, outer, 1, thickness=-1)
    cv2.circle(mask, center, inner, 0, thickness=-1)
    return mask


def rasterize_line(line: str, shape: tuple[int, int]) -> np.ndarray:
    """YOLO 라벨 한 줄을 이진 마스크로 되돌린다."""

    fields = line.split()
    points = np.array([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= shape[1]
    points[:, 1] *= shape[0]
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [points.astype(np.int32)], 1)
    return canvas


COMPONENT = {"component_id": 1, "class_name": "grasp_region", "model_class_id": 0}


class TestRingSlit(unittest.TestCase):
    def test_기본_모드에서는_고리를_제외한다(self) -> None:
        instance = make_ring_instance()
        line, audit = component_to_yolo_polygon(instance, dict(COMPONENT))
        self.assertIsNone(line)
        self.assertIn("component_not_single_hole_free_polygon", audit["reasons"])

    def test_절개선_모드는_고리를_내보내고_구멍을_보존한다(self) -> None:
        instance = make_ring_instance()
        line, audit = component_to_yolo_polygon(
            instance, dict(COMPONENT), allow_ring_slit=True
        )
        self.assertIsNotNone(line)
        self.assertTrue(audit["ring_slit"]["applied"])
        self.assertEqual(audit["ring_slit"]["holes_bridged"], 1)
        self.assertGreaterEqual(audit["polygon_iou"], 0.9)
        reconstructed = rasterize_line(line, instance.shape)
        center = instance.shape[0] // 2
        # 구멍 중심은 배경으로 남아야 한다 (contain을 몸통으로 채우지 않음).
        self.assertEqual(int(reconstructed[center, center]), 0)
        # 고리 위의 점은 채워져야 한다.
        self.assertEqual(int(reconstructed[center, center + 27]), 1)

    def test_구멍_없는_성분은_기존과_동일하게_동작한다(self) -> None:
        instance = np.zeros((64, 64), dtype=np.uint16)
        cv2.circle(instance, (32, 32), 20, 1, thickness=-1)
        line, audit = component_to_yolo_polygon(
            instance, dict(COMPONENT), allow_ring_slit=True
        )
        self.assertIsNotNone(line)
        self.assertNotIn("ring_slit", audit)

    def test_구멍_두_개도_모두_보존한다(self) -> None:
        instance = np.zeros((96, 96), dtype=np.uint16)
        cv2.circle(instance, (48, 48), 40, 1, thickness=-1)
        cv2.circle(instance, (34, 48), 9, 0, thickness=-1)
        cv2.circle(instance, (62, 48), 9, 0, thickness=-1)
        line, audit = component_to_yolo_polygon(
            instance, dict(COMPONENT), allow_ring_slit=True
        )
        self.assertIsNotNone(line)
        self.assertEqual(audit["ring_slit"]["holes_bridged"], 2)
        reconstructed = rasterize_line(line, instance.shape)
        self.assertEqual(int(reconstructed[48, 34]), 0)
        self.assertEqual(int(reconstructed[48, 62]), 0)
        self.assertEqual(int(reconstructed[48, 48]), 1)


if __name__ == "__main__":
    unittest.main()
