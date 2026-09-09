"""자체 데이터 반자동 라벨 후보의 고정 정책을 검사한다."""

from __future__ import annotations

import unittest

import numpy as np

from src.labeling.custom_pseudolabels import (
    MaskCandidate,
    combine_grasp_candidates,
    select_even_samples,
    split_whole_mug_mask,
)


class CustomPseudolabelTests(unittest.TestCase):
    def test_even_sample_selection_keeps_objects_separate(self) -> None:
        records = [
            {
                "source_id": f"mug_01/frame_{index:02d}",
                "object_id": "mug_01",
                "conversion_status": "pending_annotation",
                "is_negative": False,
            }
            for index in range(5)
        ]
        records.append(
            {
                "source_id": "background/frame_00",
                "object_id": "background_negative",
                "conversion_status": "converted",
                "is_negative": True,
            }
        )

        selected = select_even_samples(records, 3)

        self.assertEqual(
            [row["source_id"] for row in selected],
            ["mug_01/frame_00", "mug_01/frame_02", "mug_01/frame_04"],
        )

    def test_handle_and_body_remain_separate_grasp_instances(self) -> None:
        handle = np.zeros((8, 8), dtype=bool)
        handle[2:6, 1:4] = True
        body = np.zeros((8, 8), dtype=bool)
        body[1:7, 3:7] = True
        candidates = [
            MaskCandidate("body", 0.8, (3, 1, 7, 7), body),
            MaskCandidate("handle", 0.9, (1, 2, 4, 6), handle),
        ]

        semantic, instance, components, flags = combine_grasp_candidates(
            candidates,
            (8, 8),
            minimum_pixels=1,
        )

        self.assertEqual(set(np.unique(semantic)), {0, 1})
        self.assertNotIn(2, semantic)
        self.assertEqual(set(np.unique(instance)), {0, 1, 2})
        self.assertEqual([row["part"] for row in components], ["handle", "body"])
        self.assertEqual(flags, [])
        self.assertFalse(np.any((instance == 1) & (instance == 2)))

    def test_missing_handle_is_always_flagged_for_review(self) -> None:
        body = np.ones((4, 4), dtype=bool)
        _, _, components, flags = combine_grasp_candidates(
            [MaskCandidate("body", 0.7, (0, 0, 4, 4), body)],
            (4, 4),
            minimum_pixels=1,
        )

        self.assertEqual(len(components), 1)
        self.assertIn("손잡이_후보_누락", flags)

    def test_top_view_shape_split_separates_side_handle(self) -> None:
        import cv2

        whole = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(whole, (70, 60), 40, 1, thickness=-1)
        cv2.ellipse(whole, (118, 60), (28, 19), 0, 0, 360, 1, thickness=9)

        body, handle, flags, diagnostics = split_whole_mug_mask(
            whole, minimum_handle_pixels=20
        )

        self.assertFalse(np.any(body & handle))
        self.assertTrue(np.array_equal(body | handle, whole.astype(bool)))
        self.assertGreater(int(handle.sum()), 100)
        self.assertEqual(diagnostics["method"], "top_view_ellipse")
        self.assertNotIn("형태분석_손잡이_누락", flags)

    def test_side_view_shape_split_keeps_main_body(self) -> None:
        import cv2

        whole = np.zeros((160, 180), dtype=np.uint8)
        whole[25:140, 45:115] = 1
        cv2.ellipse(whole, (126, 75), (34, 42), 0, -90, 90, 1, thickness=10)

        body, handle, flags, diagnostics = split_whole_mug_mask(
            whole, minimum_handle_pixels=20
        )

        self.assertFalse(np.any(body & handle))
        self.assertTrue(np.array_equal(body | handle, whole.astype(bool)))
        self.assertGreater(int(body.sum()), int(handle.sum()))
        self.assertEqual(diagnostics["method"], "side_view_row_envelope")
        self.assertNotIn("형태분석_손잡이_누락", flags)


if __name__ == "__main__":
    unittest.main()
