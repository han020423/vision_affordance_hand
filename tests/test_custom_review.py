"""자체 마스크 수동 검수 준비 도구의 단위 테스트."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.labeling.custom_review import (
    BODY_INSTANCE_ID,
    HANDLE_INSTANCE_ID,
    inspect_mug_instance_mask,
    prepare_full_review_workspace,
    prepare_review_workspace,
    remove_tiny_instance_fragments,
    semantic_from_instances,
    standardize_mug_instances,
)


class CustomReviewTest(unittest.TestCase):
    """검수 과정에서 라벨 정책과 원본 보존이 깨지지 않는지 확인한다."""

    def test_standardize_mug_instances_uses_fixed_part_ids(self) -> None:
        """후보 ID 순서와 무관하게 손잡이=1, 몸통=2를 사용해야 한다."""

        candidate = np.array([[0, 7, 7], [4, 4, 0]], dtype=np.uint8)
        components = [
            {"instance_id": 7, "part": "body"},
            {"instance_id": 4, "part": "handle"},
        ]
        result = standardize_mug_instances(candidate, components)
        self.assertTrue(np.all(result[candidate == 4] == HANDLE_INSTANCE_ID))
        self.assertTrue(np.all(result[candidate == 7] == BODY_INSTANCE_ID))

    def test_semantic_mask_contains_only_background_and_grasp(self) -> None:
        """머그 수정본에서 functional_region 저장값 2가 생기면 안 된다."""

        instance = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        semantic = semantic_from_instances(instance)
        self.assertEqual({0, 1}, set(np.unique(semantic).tolist()))

    def test_tiny_fragments_are_removed_without_deleting_main_regions(self) -> None:
        """최소 크기 미만의 브러시 점만 제거하고 큰 후보는 유지해야 한다."""

        instance = np.zeros((20, 20), dtype=np.uint8)
        instance[2:10, 2:10] = HANDLE_INSTANCE_ID
        instance[12:20, 12:20] = BODY_INSTANCE_ID
        instance[0, 0] = HANDLE_INSTANCE_ID
        cleaned, removed = remove_tiny_instance_fragments(instance, minimum_pixels=4)
        self.assertEqual(0, int(cleaned[0, 0]))
        self.assertEqual(1, removed[HANDLE_INSTANCE_ID])
        self.assertEqual(64, int(np.count_nonzero(cleaned == HANDLE_INSTANCE_ID)))
        self.assertEqual(64, int(np.count_nonzero(cleaned == BODY_INSTANCE_ID)))

    def test_invisible_handle_is_flagged_but_body_is_kept(self) -> None:
        """손잡이가 보이지 않는 시점은 가짜 손잡이를 만들지 않고 경고만 남긴다."""

        instance = np.zeros((12, 12), dtype=np.uint8)
        instance[2:10, 3:9] = BODY_INSTANCE_ID
        components, flags = inspect_mug_instance_mask(instance, minimum_pixels=4)
        self.assertEqual(["body"], [component["part"] for component in components])
        self.assertIn("손잡이_비가시_또는_라벨없음", flags)
        self.assertNotIn("몸통_누락", flags)

    def test_prepare_workspace_preserves_candidates(self) -> None:
        """원본 후보를 수정하지 않고 별도 수정 복사본을 만들어야 한다."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "candidates"
            raw_root = root / "raw"
            output_root = root / "review"
            (candidate_root / "instance_candidates").mkdir(parents=True)
            raw_root.mkdir()

            raw_name = "WIN_20260814_00_00_00_Pro.jpg"
            review_name = f"mug_01__{raw_name}"
            image = np.full((12, 16, 3), 180, dtype=np.uint8)
            candidate = np.zeros((12, 16), dtype=np.uint8)
            candidate[2:7, 2:5] = 3
            candidate[2:10, 5:12] = 9
            cv2.imwrite(str(raw_root / raw_name), image)
            candidate_path = candidate_root / "instance_candidates" / Path(
                review_name
            ).with_suffix(".png")
            cv2.imwrite(str(candidate_path), candidate)
            row = {
                "source_id": f"mug_01/{Path(raw_name).stem}",
                "object_id": "mug_01",
                "split": "train",
                "overlay_path": f"overlays/{review_name}",
                "candidate_instance_path": candidate_path.relative_to(
                    candidate_root
                ).as_posix(),
                "components": [
                    {"instance_id": 3, "part": "handle"},
                    {"instance_id": 9, "part": "body"},
                ],
            }
            (candidate_root / "candidate_manifest.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            config_path = root / "review.yaml"
            config_path.write_text(
                "review_id: unit_test\n"
                "source_run: fixture\n"
                "reviews:\n"
                f"  {review_name}: 손잡이 누락\n",
                encoding="utf-8",
            )

            before = candidate_path.read_bytes()
            summary = prepare_review_workspace(
                config_path, candidate_root, raw_root, output_root
            )
            after = candidate_path.read_bytes()
            corrected = cv2.imread(
                str(
                    output_root
                    / "corrected_instances"
                    / Path(review_name).with_suffix(".png")
                ),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertEqual(before, after)
            self.assertEqual(1, summary["items"])
            self.assertEqual({0, 1, 2}, set(np.unique(corrected).tolist()))
            self.assertFalse(summary["functional_region_generated"])

    def test_prepare_full_workspace_copies_every_final_mask(self) -> None:
        """최종 마스크 전체를 원본과 분리된 수정 복사본으로 준비해야 한다."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_root = root / "final"
            output_root = root / "full_review"
            (final_root / "images").mkdir(parents=True)
            (final_root / "instance_masks").mkdir()
            image_name = "mug_01__frame.jpg"
            mask_name = "mug_01__frame.png"
            image = np.full((10, 14, 3), 150, dtype=np.uint8)
            mask = np.zeros((10, 14), dtype=np.uint8)
            mask[2:8, 2:6] = BODY_INSTANCE_ID
            cv2.imwrite(str(final_root / "images" / image_name), image)
            cv2.imwrite(str(final_root / "instance_masks" / mask_name), mask)
            final_row = {
                "source_id": "mug_01/frame",
                "object_id": "mug_01",
                "split": "train",
                "image_path": f"images/{image_name}",
                "instance_mask_path": f"instance_masks/{mask_name}",
                "validation_flags": ["손잡이_비가시_또는_라벨없음"],
                "annotation_status": "final_confirmation_required",
                "label_source": "human_corrected",
            }
            (final_root / "final_review_manifest.jsonl").write_text(
                json.dumps(final_row, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            summary = prepare_full_review_workspace(final_root, output_root)
            rows = [
                json.loads(line)
                for line in (output_root / "review_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]

            self.assertEqual(1, summary["items"])
            self.assertEqual("ready_for_optional_edit", rows[0]["review_status"])
            self.assertEqual("not_visible", rows[0]["handle_visibility"])
            self.assertTrue(
                (output_root / rows[0]["corrected_instance_path"]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
