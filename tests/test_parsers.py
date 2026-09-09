from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from src.datasets.affgrasp import convert_affgrasp, discover_affgrasp
from src.datasets.umd import convert_umd, deterministic_manual_subsample


class AffGraspParserTests(unittest.TestCase):
    def test_pairing_and_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 3)).save(root / "cup-sample-img.jpg")
            Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(root / "cup-sample-label.png")

            self.assertEqual(len(discover_affgrasp(root)), 1)
            records = convert_affgrasp(root, root / "out", root, dry_run=True)
            self.assertEqual(records[0].conversion_status, "excluded")
            self.assertTrue(records[0].reasons[0].startswith("size_mismatch"))

    def test_empty_mask_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 3)).save(root / "cup-sample-img.jpg")
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(root / "cup-sample-label.png")
            records = convert_affgrasp(root, root / "out", root, dry_run=True)
            self.assertEqual(records[0].conversion_status, "excluded")
            self.assertIn("empty_affordance_mask", records[0].reasons)


class UmdParserTests(unittest.TestCase):
    def test_automatic_gt_is_excluded_and_manual_gt_is_converted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_root = root / "tools" / "mug_01"
            object_root.mkdir(parents=True)
            label = np.array([[0, 1], [4, 7]], dtype=np.uint8)
            for frame, gt_type in ((1, "manual"), (2, "automatic")):
                stem = f"mug_01_{frame:08d}"
                Image.new("RGB", (2, 2)).save(object_root / f"{stem}_rgb.jpg")
                savemat(
                    object_root / f"{stem}_label.mat",
                    {"gt_label": label, "gt_type": gt_type},
                )

            records = convert_umd(
                root / "tools",
                root / "out",
                root,
                {"mug_01": "train"},
                manual_frame_modulo=None,
                dry_run=True,
            )
            self.assertEqual(records[0].conversion_status, "converted")
            self.assertIn("umd_contain_policy", records[0].reasons)
            self.assertEqual(records[1].conversion_status, "excluded")
            self.assertIn("automatic_label_excluded", records[1].reasons)

    def test_subsampling_is_exact_deterministic_and_keeps_each_object(self) -> None:
        samples = []
        quality = {}
        for object_id in ("mug_01", "mug_02"):
            for frame in range(10):
                source_id = f"{object_id}/{object_id}_{frame:08d}"
                samples.append(
                    (
                        source_id,
                        Path("tools") / object_id / f"{object_id}_{frame:08d}_rgb.jpg",
                        Path("tools") / object_id / f"{object_id}_{frame:08d}_label.mat",
                    )
                )
                quality[source_id] = "manual"
        first = deterministic_manual_subsample(
            samples, quality, 8, manual_frame_modulo=None
        )
        second = deterministic_manual_subsample(
            samples, quality, 8, manual_frame_modulo=None
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual({source_id.split("/")[0] for source_id in first}, {"mug_01", "mug_02"})


if __name__ == "__main__":
    unittest.main()
