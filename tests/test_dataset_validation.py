from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.datasets.splits import (
    build_umd_object_splits,
    find_split_leakage,
    load_object_splits,
    parse_umd_official_folds,
)
import numpy as np

from src.datasets.umd import frame_number, ground_truth_type


class DatasetValidationTests(unittest.TestCase):
    def test_split_leakage_is_detected_by_object_id(self) -> None:
        records = [
            {"object_id": "mug_01", "split": "train"},
            {"object_id": "mug_01", "split": "test"},
            {"object_id": "mug_02", "split": "validation"},
        ]
        self.assertEqual(find_split_leakage(records), {"mug_01": ["test", "train"]})

    def test_duplicate_split_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.yaml"
            path.write_text(
                "splits:\n  train: [mug_01]\n  test: [mug_01]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "분할에 모두 포함"):
                load_object_splits(path)

    def test_manual_ground_truth_metadata(self) -> None:
        self.assertEqual(frame_number("mug_01/mug_01_000003"), 3)
        self.assertEqual(ground_truth_type({"gt_type": np.array(["manual"])}), "manual")
        self.assertEqual(ground_truth_type({"gt_type": np.array(["automatic"])}), "automatic")
        self.assertIsNone(ground_truth_type({}))

    def test_official_fold_split_is_object_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.txt"
            path.write_text(
                "folds: 1 0 0 0 0 0 0 0 0 0    object: mug_01\n"
                "folds: 0 1 0 0 0 0 0 0 0 0    object: mug_02\n"
                "folds: 0 0 1 0 0 0 0 0 0 0    object: mug_03\n",
                encoding="utf-8",
            )
            memberships = parse_umd_official_folds(path)
            self.assertEqual(memberships["mug_01"], frozenset({0}))
            splits = build_umd_object_splits(
                path,
                {"mug_01", "mug_02", "mug_03"},
                validation_fold=1,
                test_fold=0,
            )
            self.assertEqual(
                splits,
                {"mug_01": "test", "mug_02": "validation", "mug_03": "train"},
            )


if __name__ == "__main__":
    unittest.main()
