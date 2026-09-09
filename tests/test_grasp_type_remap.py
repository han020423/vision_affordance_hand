"""파지 종류 3클래스 재매핑 규칙 단위 테스트."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.yolo_export_grasp_type import (  # noqa: E402
    custom_component_grasp_type,
    remap_label_line,
    umd_component_grasp_type,
)


class TestRemapLabelLine(unittest.TestCase):
    def test_클래스_토큰만_바뀐다(self) -> None:
        line = "0 0.10000000 0.20000000 0.30000000 0.40000000 0.50000000 0.60000000"
        remapped = remap_label_line(line, 2)
        self.assertTrue(remapped.startswith("2 "))
        self.assertEqual(remapped.split()[1:], line.split()[1:])

    def test_좌표가_부족하면_실패한다(self) -> None:
        with self.assertRaises(ValueError):
            remap_label_line("0 0.1 0.2", 1)


class TestUmdVote(unittest.TestCase):
    def setUp(self) -> None:
        # 4x4 인스턴스 마스크: 컴포넌트 1이 왼쪽 절반을 차지한다.
        self.instance = np.zeros((4, 4), dtype=np.uint16)
        self.instance[:, :2] = 1

    def test_grasp_다수는_handle이_된다(self) -> None:
        raw = np.zeros((4, 4), dtype=np.uint8)
        raw[:, 0] = 1  # grasp 4픽셀
        raw[0, 1] = 7  # wrap-grasp 1픽셀
        new_class, audit = umd_component_grasp_type(raw, self.instance, 1)
        self.assertEqual(new_class, 0)
        self.assertEqual(audit["handle_votes"], 4)
        self.assertEqual(audit["body_votes"], 1)
        self.assertFalse(audit["ambiguous"])

    def test_wrap_grasp_다수는_body가_된다(self) -> None:
        raw = np.zeros((4, 4), dtype=np.uint8)
        raw[:, :2] = 7
        new_class, audit = umd_component_grasp_type(raw, self.instance, 1)
        self.assertEqual(new_class, 1)
        self.assertEqual(audit["body_votes"], 8)

    def test_근소한_다수결은_모호_표시가_남는다(self) -> None:
        raw = np.zeros((4, 4), dtype=np.uint8)
        raw[:2, :2] = 1  # handle 4
        raw[2:, :2] = 7  # body 4 → 동률이면 handle 우선, 소수 비율 0.5로 모호
        new_class, audit = umd_component_grasp_type(raw, self.instance, 1)
        self.assertEqual(new_class, 0)
        self.assertTrue(audit["ambiguous"])

    def test_파지_픽셀이_없으면_제외한다(self) -> None:
        raw = np.zeros((4, 4), dtype=np.uint8)
        new_class, audit = umd_component_grasp_type(raw, self.instance, 1)
        self.assertIsNone(new_class)
        self.assertEqual(audit["reason"], "no_grasp_source_pixels")

    def test_크기_불일치는_제외한다(self) -> None:
        raw = np.zeros((3, 3), dtype=np.uint8)
        new_class, audit = umd_component_grasp_type(raw, self.instance, 1)
        self.assertIsNone(new_class)
        self.assertEqual(audit["reason"], "raw_instance_size_mismatch")


class TestCustomPart(unittest.TestCase):
    def test_handle은_클래스_0이다(self) -> None:
        new_class, _ = custom_component_grasp_type({"part": "handle"})
        self.assertEqual(new_class, 0)

    def test_body는_클래스_1이다(self) -> None:
        new_class, _ = custom_component_grasp_type({"part": "body"})
        self.assertEqual(new_class, 1)

    def test_part가_없으면_제외한다(self) -> None:
        new_class, audit = custom_component_grasp_type({})
        self.assertIsNone(new_class)
        self.assertEqual(audit["reason"], "unknown_custom_part")


if __name__ == "__main__":
    unittest.main()
