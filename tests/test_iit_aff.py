"""IIT-AFF 라벨 매핑과 행렬 파서 단위 테스트."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.iit_aff import (  # noqa: E402
    convert_iit_mask,
    load_iit_label_matrix,
)
from src.labeling.policy import (  # noqa: E402
    FUNCTIONAL_STORAGE_VALUE,
    GRASP_STORAGE_VALUE,
    IGNORE_VALUE,
)


class TestConvertIitMask(unittest.TestCase):
    def test_grasp와_wrap은_grasp_저장값이_된다(self) -> None:
        source = np.array([[5, 9], [0, 0]], dtype=np.int16)
        converted, summary = convert_iit_mask(source)
        self.assertEqual(int(converted[0, 0]), GRASP_STORAGE_VALUE)
        self.assertEqual(int(converted[0, 1]), GRASP_STORAGE_VALUE)
        self.assertEqual(int(converted[1, 0]), 0)
        self.assertIn("grasp", summary.original_labels)
        self.assertIn("w-grasp", summary.original_labels)

    def test_기능_라벨은_functional이_된다(self) -> None:
        source = np.array([[2, 3, 4], [6, 7, 8]], dtype=np.int16)
        converted, _ = convert_iit_mask(source)
        self.assertTrue(np.all(converted == FUNCTIONAL_STORAGE_VALUE))

    def test_contain은_ignore와_정책_사유가_된다(self) -> None:
        source = np.array([[1, 0]], dtype=np.int16)
        converted, summary = convert_iit_mask(source)
        self.assertEqual(int(converted[0, 0]), IGNORE_VALUE)
        self.assertIn("iit_contain_policy", summary.ignore_reasons)

    def test_알_수_없는_값은_ignore와_검토_사유가_된다(self) -> None:
        source = np.array([[42, 0]], dtype=np.int16)
        converted, summary = convert_iit_mask(source)
        self.assertEqual(int(converted[0, 0]), IGNORE_VALUE)
        self.assertEqual(summary.unknown_source_values, (42,))
        self.assertIn("unknown_iit_value", summary.ignore_reasons)


class TestLoadIitLabelMatrix(unittest.TestCase):
    def test_정상_행렬을_읽는다(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "label.txt"
            path.write_text("0 5 9\n1 2 0\n", encoding="ascii")
            matrix = load_iit_label_matrix(path)
            self.assertEqual(matrix.shape, (2, 3))
            self.assertEqual(int(matrix[0, 1]), 5)
            self.assertEqual(int(matrix[1, 0]), 1)

    def test_행_길이가_다르면_실패한다(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "label.txt"
            path.write_text("0 5 9\n1 2\n", encoding="ascii")
            with self.assertRaises(ValueError):
                load_iit_label_matrix(path)


if __name__ == "__main__":
    unittest.main()
