"""전체 검수 자체 데이터를 승인 세트로 승격하는 테스트."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.labeling.custom_approval import promote_full_review


class CustomApprovalTest(unittest.TestCase):
    """원본 보존, 배경 포함, 고정 라벨 정책을 검증한다."""

    def test_promote_full_review_includes_object_and_negative(self) -> None:
        """검수 객체와 검증 배경을 함께 승인 manifest로 만들어야 한다."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            raw_root = root / "raw"
            output_root = root / "approved"
            (workspace / "corrected_instances").mkdir(parents=True)
            raw_root.mkdir()

            object_image = raw_root / "object.jpg"
            background_image = raw_root / "background.jpg"
            image = np.full((16, 20, 3), 120, dtype=np.uint8)
            cv2.imwrite(str(object_image), image)
            cv2.imwrite(str(background_image), image)
            instance = np.zeros((16, 20), dtype=np.uint8)
            instance[2:8, 2:7] = 1
            instance[4:14, 8:18] = 2
            instance[0, 0] = 1
            corrected_path = workspace / "corrected_instances" / "mug_01__object.png"
            cv2.imwrite(str(corrected_path), instance)
            review = {
                "source_id": "mug_01/object",
                "object_id": "mug_01",
                "split": "train",
                "corrected_instance_path": "corrected_instances/mug_01__object.png",
                "review_status": "human_corrected",
                "handle_visibility": "visible",
            }
            (workspace / "review_manifest.jsonl").write_text(
                json.dumps(review) + "\n", encoding="utf-8"
            )
            sources = [
                {
                    "source_dataset": "custom",
                    "source_id": "mug_01/object",
                    "object_id": "mug_01",
                    "object_id_status": "verified_physical_object",
                    "image_path": str(object_image),
                    "source_mask_path": "",
                    "semantic_mask_path": "",
                    "instance_mask_path": "",
                    "original_labels": ["unlabeled_custom_capture"],
                    "mapped_labels": [],
                    "split": "train",
                    "conversion_status": "pending_annotation",
                    "is_negative": False,
                },
                {
                    "source_dataset": "custom",
                    "source_id": "background_negative/background",
                    "object_id": "background_negative",
                    "object_id_status": "no_physical_object_negative",
                    "image_path": str(background_image),
                    "source_mask_path": "",
                    "semantic_mask_path": "",
                    "instance_mask_path": "",
                    "original_labels": [],
                    "mapped_labels": ["background"],
                    "split": "train",
                    "conversion_status": "converted",
                    "is_negative": True,
                },
            ]
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
            )

            summary = promote_full_review(
                workspace,
                source_manifest,
                output_root,
                root,
                minimum_fragment_pixels=4,
            )
            approved = [
                json.loads(line)
                for line in (output_root / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            object_row, negative_row = approved
            object_instance = cv2.imread(
                str(root / object_row["instance_mask_path"]), cv2.IMREAD_UNCHANGED
            )
            negative_semantic = cv2.imread(
                str(root / negative_row["semantic_mask_path"]), cv2.IMREAD_UNCHANGED
            )

            self.assertEqual(2, summary["records"])
            self.assertEqual(1, summary["object_images"])
            self.assertEqual(1, summary["background_images"])
            self.assertEqual("human_verified", object_row["annotation_status"])
            self.assertEqual("converted", object_row["conversion_status"])
            self.assertEqual(["grasp_region"], object_row["mapped_labels"])
            self.assertEqual(0, int(object_instance[0, 0]))
            self.assertTrue(np.all(negative_semantic == 0))
            self.assertFalse(object_row["functional_region_generated"])


if __name__ == "__main__":
    unittest.main()
