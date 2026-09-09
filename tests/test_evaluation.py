from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_test_seg import binary_scores, polygons_to_masks, safe_ratio, summarize_formal_rows


def test_safe_ratio_preserves_undefined_empty_case() -> None:
    assert safe_ratio(0, 0) is None
    assert safe_ratio(1, 4) == 0.25


def test_binary_scores_report_miss_and_overlap() -> None:
    ground_truth = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    scores = binary_scores(ground_truth, prediction)
    assert scores["intersection_pixels"] == 1
    assert scores["iou"] == 1 / 3
    assert scores["pixel_precision"] == 0.5
    assert scores["pixel_recall"] == 0.5
    assert scores["miss_ratio"] == 0.5


def test_polygons_to_masks_keeps_both_project_classes(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(
        "0 0.0 0.0 0.4 0.0 0.4 0.4 0.0 0.4\n"
        "1 0.6 0.6 1.0 0.6 1.0 1.0 0.6 1.0\n",
        encoding="utf-8",
    )
    masks = polygons_to_masks(label, width=10, height=10)
    assert np.count_nonzero(masks[0]) > 0
    assert np.count_nonzero(masks[1]) > 0
    assert not np.any(masks[0] & masks[1])


def test_summarize_formal_rows_preserves_missing_values() -> None:
    row = {
        "image": "sample.jpg",
        "object_id": "umd__cup_03",
        "overlay_path": "sample_overlay.jpg",
        "review_error_score": 0.5,
        "grasp_gt_pixels": 10,
        "grasp_pred_pixels": 0,
        "grasp_iou": 0.0,
        "grasp_miss_ratio": 1.0,
        "functional_gt_pixels": 0,
        "functional_pred_pixels": 5,
        "functional_iou": None,
        "functional_miss_ratio": None,
        "pred_functional_on_gt_grasp_ratio": 0.2,
        "pred_grasp_on_gt_functional_ratio": None,
    }

    summary = summarize_formal_rows([row])

    assert summary["classes"]["grasp_region"]["prediction_absent_images"] == 1
    assert summary["classes"]["functional_region"]["ground_truth_present_images"] == 0
    assert summary["by_object"]["umd__cup_03"]["functional_region"]["mean_pixel_iou"] is None
    assert (
        summary["class_overlap_review"][
            "predicted_functional_overlapping_gt_grasp_over_0_1_images"
        ]
        == 1
    )
