"""RF-DETR → 파지 후보 선택 어댑터 단위 테스트.

`scripts/run_realtime_rfdetr.py::detections_to_arrays`가 rfdetr 예측을
`extract_candidates_from_arrays` 규약(0=handle, 1=body, 2=functional)에 맞는
배열로 바꾸는지 검증한다. rfdetr 패키지 없이 실행할 수 있도록 mask/class_id/
confidence 속성만 가진 가짜 객체(덕 타이핑)를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
for entry in (str(REPOSITORY_ROOT), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_realtime_rfdetr import detections_to_arrays  # noqa: E402
from src.grasp_selection import extract_candidates_from_arrays  # noqa: E402

HEIGHT, WIDTH = 120, 160
FRAME_SHAPE = (HEIGHT, WIDTH, 3)
# 설정 기본값과 같은 후보 추출 조건 (configs/grasp_selection.yaml candidate 절).
EXTRACT_KWARGS = {"min_confidence": 0.35, "min_area_px": 400, "boundary_margin_px": 4}


@dataclass
class FakeDetections:
    """rfdetr(supervision Detections)의 관심 속성만 흉내 내는 객체."""

    mask: object
    class_id: object
    confidence: object


def rect_mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """지정 사각형만 참인 (H, W) bool 마스크를 만든다."""

    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def three_class_detections(base: int = 0, handle_confidence: float = 0.9) -> FakeDetections:
    """손잡이/몸통/기능 사각형 세 개짜리 합성 검출을 만든다.

    사각형은 min_area_px=400을 넘는 크기로 잡고 프레임 경계에서 떨어뜨린다.
    """

    masks = np.stack(
        [
            rect_mask(10, 40, 10, 40),     # handle: 30x30 = 900px
            rect_mask(60, 100, 60, 120),   # body:   40x60 = 2400px
            rect_mask(10, 40, 110, 150),   # functional: 30x40 = 1200px
        ]
    )
    class_ids = np.array([0, 1, 2], dtype=int) + base
    confidences = np.array([handle_confidence, 0.8, 0.7], dtype=float)
    return FakeDetections(mask=masks, class_id=class_ids, confidence=confidences)


class TestDetectionsToArrays(unittest.TestCase):
    """변환 함수 자체의 동작."""

    def test_기본_변환과_형식(self) -> None:
        masks, class_ids, confidences = detections_to_arrays(three_class_detections())
        self.assertEqual(masks.shape, (3, HEIGHT, WIDTH))
        self.assertEqual(masks.dtype, np.bool_)
        self.assertEqual(class_ids.tolist(), [0, 1, 2])
        self.assertEqual(len(confidences), 3)

    def test_class_id_base_1이_0기준으로_정규화(self) -> None:
        detections = three_class_detections(base=1)  # COCO식 1/2/3
        _, class_ids, _ = detections_to_arrays(detections, class_id_base=1)
        self.assertEqual(class_ids.tolist(), [0, 1, 2])

    def test_None과_빈_검출은_빈_배열(self) -> None:
        for detections in (
            None,
            FakeDetections(mask=None, class_id=np.array([0]), confidence=np.array([0.9])),
            FakeDetections(mask=np.zeros((0, HEIGHT, WIDTH), dtype=bool),
                           class_id=np.zeros((0,), dtype=int),
                           confidence=np.zeros((0,), dtype=float)),
        ):
            masks, class_ids, confidences = detections_to_arrays(detections)
            self.assertEqual(len(masks), 0)
            self.assertEqual(len(class_ids), 0)
            self.assertEqual(len(confidences), 0)

    def test_길이_불일치는_오류(self) -> None:
        broken = FakeDetections(
            mask=np.zeros((2, HEIGHT, WIDTH), dtype=bool),
            class_id=np.array([0], dtype=int),
            confidence=np.array([0.9, 0.8], dtype=float),
        )
        with self.assertRaises(ValueError):
            detections_to_arrays(broken)


class TestAdapterToCandidates(unittest.TestCase):
    """변환 함수 → extract_candidates_from_arrays 전체 경로."""

    def test_합성_3클래스에서_후보2와_회피마스크(self) -> None:
        detections = three_class_detections()
        masks, class_ids, confidences = detections_to_arrays(detections)
        candidates, functional_mask = extract_candidates_from_arrays(
            masks, class_ids, confidences, FRAME_SHAPE, **EXTRACT_KWARGS
        )
        self.assertEqual(sorted(c.class_id for c in candidates), [0, 1])
        # 기능 영역은 후보가 아니라 회피 마스크로 합쳐진다.
        expected_functional = rect_mask(10, 40, 110, 150)
        np.testing.assert_array_equal(functional_mask, expected_functional)
        # 파지점은 각 후보 마스크 안쪽에 있어야 한다.
        for candidate in candidates:
            x, y = candidate.grasp_point
            self.assertTrue(candidate.mask[y, x], f"{candidate.class_name} 파지점이 마스크 밖")

    def test_base1_입력도_같은_후보를_만든다(self) -> None:
        detections = three_class_detections(base=1)
        masks, class_ids, confidences = detections_to_arrays(detections, class_id_base=1)
        candidates, functional_mask = extract_candidates_from_arrays(
            masks, class_ids, confidences, FRAME_SHAPE, **EXTRACT_KWARGS
        )
        self.assertEqual(sorted(c.class_id for c in candidates), [0, 1])
        self.assertTrue(functional_mask.any())

    def test_빈_검출은_빈_후보(self) -> None:
        masks, class_ids, confidences = detections_to_arrays(None)
        candidates, functional_mask = extract_candidates_from_arrays(
            masks, class_ids, confidences, FRAME_SHAPE, **EXTRACT_KWARGS
        )
        self.assertEqual(candidates, [])
        self.assertFalse(functional_mask.any())

    def test_저신뢰_손_오검출은_걸러진다(self) -> None:
        # 2026-08-24 실측에서 맨손이 handle 0.25로 잠깐 오검출된 회귀 사례:
        # min_confidence 0.35가 이런 저신뢰 후보를 걸러야 한다.
        detections = three_class_detections(handle_confidence=0.25)
        masks, class_ids, confidences = detections_to_arrays(detections)
        candidates, _ = extract_candidates_from_arrays(
            masks, class_ids, confidences, FRAME_SHAPE, **EXTRACT_KWARGS
        )
        self.assertEqual([c.class_id for c in candidates], [1], "body만 남아야 한다")


if __name__ == "__main__":
    unittest.main()
