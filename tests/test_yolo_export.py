from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.datasets.yolo_export import (
    can_apply_contain_background_override,
    component_to_yolo_polygon,
    export_public_yolo,
)


class YoloExportTests(unittest.TestCase):
    @staticmethod
    def _write_ignore_fixture(
        root: Path,
        *,
        reasons: list[str],
        original_labels: list[str],
    ) -> tuple[Path, Path]:
        """ignore와 grasp가 함께 있는 최소 manifest를 만든다."""

        image_path = root / "image.jpg"
        semantic_path = root / "semantic.png"
        instance_path = root / "instance.png"
        Image.new("RGB", (8, 8)).save(image_path)
        semantic = np.zeros((8, 8), dtype=np.uint8)
        semantic[1:4, 1:4] = 1
        semantic[4:7, 4:7] = 255
        instance = np.zeros((8, 8), dtype=np.uint16)
        instance[1:4, 1:4] = 1
        Image.fromarray(semantic).save(semantic_path)
        cv2.imwrite(str(instance_path), instance)
        record = {
            "source_dataset": "umd",
            "source_id": "mug_01/frame",
            "object_id": "mug_01",
            "split": "train",
            "conversion_status": "converted",
            "image_path": str(image_path),
            "semantic_mask_path": str(semantic_path),
            "instance_mask_path": str(instance_path),
            "original_labels": original_labels,
            "mapped_labels": ["background", "grasp_region", "ignore"],
            "reasons": reasons,
            "components": [
                {"component_id": 1, "class_name": "grasp_region", "model_class_id": 0}
            ],
        }
        manifest_path = root / "manifest.jsonl"
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return manifest_path, semantic_path

    def test_disconnected_components_remain_separate_polygons(self) -> None:
        instances = np.zeros((8, 8), dtype=np.uint16)
        instances[1:4, 1:4] = 1
        instances[4:7, 4:7] = 2
        components = [
            {"component_id": 1, "class_name": "grasp_region", "model_class_id": 0},
            {"component_id": 2, "class_name": "grasp_region", "model_class_id": 0},
        ]
        lines = [component_to_yolo_polygon(instances, component)[0] for component in components]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line is not None and line.startswith("0 ") for line in lines))

    def test_small_topology_artifacts_can_be_cleaned_for_yolo_only(self) -> None:
        """작은 구멍·고립 조각 정리는 원본 배열을 바꾸지 않고 기록되어야 한다."""

        instances = np.zeros((16, 16), dtype=np.uint16)
        instances[3:14, 3:13] = 1
        instances[5:7, 5:7] = 0
        instances[0:2, 14:16] = 1
        original = instances.copy()
        component = {
            "component_id": 1,
            "class_name": "grasp_region",
            "model_class_id": 0,
        }

        strict_line, _ = component_to_yolo_polygon(instances, component)
        cleaned_line, audit = component_to_yolo_polygon(
            instances,
            component,
            max_secondary_contour_area=8,
            max_hole_contour_area=8,
        )

        self.assertIsNone(strict_line)
        self.assertIsNotNone(cleaned_line)
        self.assertTrue(audit["topology_cleanup"]["applied"])
        self.assertGreater(audit["topology_cleanup"]["removed_pixels"], 0)
        self.assertGreater(audit["topology_cleanup"]["filled_pixels"], 0)
        np.testing.assert_array_equal(instances, original)

    def test_ignore_sample_is_logged_and_not_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = self._write_ignore_fixture(
                root,
                reasons=["umd_contain_policy"],
                original_labels=["background", "contain", "wrap-grasp"],
            )
            audits, summary = export_public_yolo(
                [manifest_path], root / "output", root, dry_run=True
            )
            self.assertEqual(audits[0]["export_status"], "excluded")
            self.assertIn("ignore_pixels_not_supported_by_yolo_polygon", audits[0]["reasons"])
            self.assertEqual(summary["counts"]["excluded"], 1)

    def test_contain_experiment_keeps_grasp_and_preserves_source_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, semantic_path = self._write_ignore_fixture(
                root,
                reasons=["umd_contain_policy"],
                original_labels=["background", "contain", "wrap-grasp"],
            )
            original_bytes = semantic_path.read_bytes()
            audits, summary = export_public_yolo(
                [manifest_path],
                root / "output",
                root,
                ignore_export_policy="contain_as_background_keep_grasp",
                dry_run=False,
            )

            label = (root / "output" / audits[0]["exports"][0]["label_path"]).read_text(
                encoding="utf-8"
            )
            self.assertEqual(audits[0]["export_status"], "exported")
            self.assertEqual(audits[0]["yolo_training_override"]["status"], "applied")
            self.assertTrue(label.startswith("0 "))
            self.assertNotIn("\n1 ", "\n" + label)
            self.assertEqual(semantic_path.read_bytes(), original_bytes)
            self.assertEqual(summary["counts"]["contain_background_override_applied"], 1)

    def test_unknown_ignore_reason_is_still_excluded_in_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = self._write_ignore_fixture(
                root,
                reasons=["unknown_label"],
                original_labels=["background", "unknown", "wrap-grasp"],
            )
            audits, _ = export_public_yolo(
                [manifest_path],
                root / "output",
                root,
                ignore_export_policy="contain_as_background_keep_grasp",
                dry_run=True,
            )
            self.assertEqual(audits[0]["export_status"], "excluded")
            self.assertIn("ignore_pixels_not_supported_by_yolo_polygon", audits[0]["reasons"])

    def test_override_requires_exact_contain_policy_evidence(self) -> None:
        valid = {
            "source_dataset": "umd",
            "conversion_status": "converted",
            "original_labels": ["background", "contain", "wrap-grasp"],
            "mapped_labels": ["background", "grasp_region", "ignore"],
            "reasons": ["umd_contain_policy"],
        }
        self.assertTrue(can_apply_contain_background_override(valid))
        valid["reasons"] = ["umd_contain_policy", "label_conflict"]
        self.assertFalse(can_apply_contain_background_override(valid))

    def test_verified_negative_is_exported_with_empty_label(self) -> None:
        """사람이 확인한 배경 영상은 0바이트 YOLO 라벨로 포함해야 한다."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "background.jpg"
            semantic_path = root / "semantic.png"
            instance_path = root / "instance.png"
            Image.new("RGB", (8, 8)).save(image_path)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(semantic_path)
            cv2.imwrite(str(instance_path), np.zeros((8, 8), dtype=np.uint16))
            record = {
                "source_dataset": "custom",
                "source_id": "background/bg_001",
                "object_id": "background_negative",
                "split": "train",
                "conversion_status": "converted",
                "annotation_status": "verified_empty",
                "is_negative": True,
                "image_path": str(image_path),
                "semantic_mask_path": str(semantic_path),
                "instance_mask_path": str(instance_path),
                "original_labels": ["background"],
                "mapped_labels": ["background"],
                "components": [],
            }
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            audits, summary = export_public_yolo(
                [manifest_path], root / "output", root, dry_run=False
            )

            label_path = root / "output" / audits[0]["exports"][0]["label_path"]
            self.assertEqual(audits[0]["export_status"], "exported_negative")
            self.assertEqual(label_path.read_bytes(), b"")
            self.assertEqual(summary["counts"]["verified_negative_samples_exported"], 1)


if __name__ == "__main__":
    unittest.main()
