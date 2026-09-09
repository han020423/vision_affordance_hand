from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.custom import parse_capture_timestamp, prepare_custom_dataset


class CustomDatasetTests(unittest.TestCase):
    def test_windows_camera_timestamp_accepts_duplicate_suffix(self) -> None:
        path = Path("WIN_20260814_17_47_47_Pro (2).jpg")
        timestamp = parse_capture_timestamp(path)
        self.assertIsNotNone(timestamp)
        self.assertEqual(timestamp.isoformat(), "2026-08-14T17:47:47")

    def test_preparation_preserves_raw_and_builds_empty_negative_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/custom/images"
            raw.mkdir(parents=True)
            object_image = raw / "WIN_20260814_10_00_00_Pro.jpg"
            negative_image = raw / "WIN_20260814_10_01_00_Pro.jpg"
            Image.new("RGB", (12, 8), "red").save(object_image)
            Image.new("RGB", (12, 8), "white").save(negative_image)
            raw_before = {path.name: path.read_bytes() for path in raw.iterdir()}

            config = root / "custom.yaml"
            config.write_text(
                """raw_image_root: data/raw/custom/images
labeling_output_root: data/interim/custom_labeling
manifest_path: data/interim/custom.jsonl
strict_unassigned: true
classes: {0: grasp_region, 1: functional_region}
ignore_index: 255
reserved_test_object_id: mug_04
captures:
  - capture_id: object
    object_id: mug_01
    category: mug
    split: train
    scene: desk
    start: '2026-08-14T10:00:00'
    end: '2026-08-14T10:00:00'
    expected_count: 1
  - capture_id: negative
    object_id: background_negative
    category: background
    split: train
    scene: desk
    start: '2026-08-14T10:01:00'
    end: '2026-08-14T10:01:00'
    expected_count: 1
    is_negative: true
""",
                encoding="utf-8",
            )

            records, summary = prepare_custom_dataset(config, root)

            self.assertEqual(summary["source_image_count"], 2)
            by_object = {record.object_id: record for record in records}
            self.assertEqual(by_object["mug_01"].conversion_status, "pending_annotation")
            self.assertEqual(by_object["background_negative"].conversion_status, "converted")
            semantic = np.asarray(
                Image.open(root / str(by_object["background_negative"].semantic_mask_path))
            )
            self.assertEqual(semantic.shape, (8, 12))
            self.assertTrue(np.all(semantic == 0))
            self.assertEqual(
                raw_before,
                {path.name: path.read_bytes() for path in raw.iterdir()},
            )

    def test_unassigned_image_is_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/custom/images"
            raw.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(raw / "WIN_20260814_11_00_00_Pro.jpg")
            config = root / "custom.yaml"
            config.write_text(
                """raw_image_root: data/raw/custom/images
labeling_output_root: data/interim/custom_labeling
manifest_path: data/interim/custom.jsonl
strict_unassigned: true
classes: {0: grasp_region, 1: functional_region}
ignore_index: 255
captures:
  - capture_id: expected
    object_id: mug_01
    category: mug
    split: train
    scene: desk
    start: '2026-08-14T10:00:00'
    end: '2026-08-14T10:00:00'
    expected_count: 1
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "추측해서 배정하지 않습니다"):
                prepare_custom_dataset(config, root, dry_run=True)


if __name__ == "__main__":
    unittest.main()

