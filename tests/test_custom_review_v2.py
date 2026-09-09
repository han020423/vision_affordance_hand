"""3클래스(functional 포함) 검수 도구 확장의 단위 테스트."""

from __future__ import annotations

import unittest

import numpy as np

from src.labeling.custom_review import (
    BODY_INSTANCE_ID,
    FUNCTIONAL_INSTANCE_ID,
    HANDLE_INSTANCE_ID,
    semantic_from_instances,
    standardize_category_instances,
    validate_review_masks,
)


class SemanticFromInstancesTest(unittest.TestCase):
    def test_기능_인스턴스는_functional_값이_된다(self) -> None:
        instance = np.zeros((6, 6), dtype=np.uint8)
        instance[0, 0] = HANDLE_INSTANCE_ID
        instance[1, 1] = BODY_INSTANCE_ID
        instance[2, 2] = FUNCTIONAL_INSTANCE_ID
        semantic = semantic_from_instances(instance)
        self.assertEqual(int(semantic[0, 0]), 1)
        self.assertEqual(int(semantic[1, 1]), 1)
        self.assertEqual(int(semantic[2, 2]), 2)
        self.assertEqual(int(semantic[3, 3]), 0)


class StandardizeCategoryInstancesTest(unittest.TestCase):
    def test_가위_고리_두_개는_같은_손잡이_값을_공유한다(self) -> None:
        candidate = np.zeros((8, 8), dtype=np.uint8)
        candidate[0, 0] = 1   # 고리 1
        candidate[1, 1] = 2   # 고리 2
        candidate[2, 2] = 3   # 날
        components = [
            {"instance_id": 1, "part": "handle", "label": "grasp_region"},
            {"instance_id": 2, "part": "handle", "label": "grasp_region"},
            {"instance_id": 3, "part": "blade", "label": "functional_region"},
        ]
        standardized = standardize_category_instances(candidate, components)
        self.assertEqual(int(standardized[0, 0]), HANDLE_INSTANCE_ID)
        self.assertEqual(int(standardized[1, 1]), HANDLE_INSTANCE_ID)
        self.assertEqual(int(standardized[2, 2]), FUNCTIONAL_INSTANCE_ID)

    def test_머그_몸통은_기존_값을_유지한다(self) -> None:
        candidate = np.zeros((4, 4), dtype=np.uint8)
        candidate[0, 0] = 1
        candidate[1, 1] = 2
        components = [
            {"instance_id": 1, "part": "handle", "label": "grasp_region"},
            {"instance_id": 2, "part": "body", "label": "grasp_region"},
        ]
        standardized = standardize_category_instances(candidate, components)
        self.assertEqual(int(standardized[0, 0]), HANDLE_INSTANCE_ID)
        self.assertEqual(int(standardized[1, 1]), BODY_INSTANCE_ID)

    def test_알_수_없는_라벨은_거부한다(self) -> None:
        candidate = np.ones((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            standardize_category_instances(
                candidate, [{"instance_id": 1, "part": "?", "label": "ignore"}]
            )


class ValidateReviewMasksTest(unittest.TestCase):
    def test_머그는_기능_부위를_금지한다(self) -> None:
        problems = validate_review_masks(
            "mug", handle_pixels=10, body_pixels=10, functional_pixels=5,
            handle_not_visible=False,
        )
        self.assertTrue(any("기능 부위" in p for p in problems))

    def test_머그_정상_저장은_문제없음(self) -> None:
        self.assertEqual(
            validate_review_masks(
                "mug", handle_pixels=10, body_pixels=10, functional_pixels=0,
                handle_not_visible=False,
            ),
            [],
        )

    def test_가위는_기능_부위가_필수다(self) -> None:
        problems = validate_review_masks(
            "scissors", handle_pixels=10, body_pixels=0, functional_pixels=0,
            handle_not_visible=False,
        )
        self.assertTrue(any("기능 부위" in p for p in problems))

    def test_가위에_몸통을_칠하면_거부한다(self) -> None:
        problems = validate_review_masks(
            "scissors", handle_pixels=10, body_pixels=3, functional_pixels=10,
            handle_not_visible=False,
        )
        self.assertTrue(any("몸통" in p for p in problems))

    def test_드라이버_정상_저장은_문제없음(self) -> None:
        self.assertEqual(
            validate_review_masks(
                "screwdriver", handle_pixels=10, body_pixels=0, functional_pixels=10,
                handle_not_visible=False,
            ),
            [],
        )

    def test_정의되지_않은_카테고리는_거부한다(self) -> None:
        problems = validate_review_masks(
            "bottle", handle_pixels=1, body_pixels=1, functional_pixels=0,
            handle_not_visible=False,
        )
        self.assertTrue(any("정의되지 않은" in p for p in problems))

    def test_손잡이_비가시_규칙은_모든_카테고리에_적용된다(self) -> None:
        problems = validate_review_masks(
            "screwdriver", handle_pixels=0, body_pixels=0, functional_pixels=10,
            handle_not_visible=False,
        )
        self.assertTrue(any("손잡이" in p for p in problems))
        self.assertEqual(
            validate_review_masks(
                "screwdriver", handle_pixels=0, body_pixels=0, functional_pixels=10,
                handle_not_visible=True,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
