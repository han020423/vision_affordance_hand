#!/usr/bin/env python
"""고정 신뢰도 픽셀 IoU 검수: YOLO(H)와 RF-DETR(J)를 같은 GT로 비교한다.

기존 evaluate_test_seg.py의 고정 신뢰도 진단 프로토콜을 따른다.

- 고정 confidence 0.25, 클래스별 예측 합집합 마스크 대 GT 합집합 마스크
- IoU 정의는 binary_scores와 동일(합집합 0이면 null)
- GT는 실험 K COCO 변환본의 valid 어노테이션을 공용으로 사용한다
  (두 모델 모두 같은 원본 승인 마스크에서 나온 같은 폴리곤으로 평가)
- 자체 val(mug_03)과 공개 val을 절대 섞어 보고하지 않고 분리 집계한다

백엔드 두 개를 지원하며 각자의 conda 환경에서 실행한다.

  ultralytics: 실험 H best.pt        (hjh_vision_hand_train 환경)
  rfdetr:      실험 K 체크포인트     (hjh_rfdetr 환경)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# COCO category_id(1시작) -> 모델 클래스 ID(0시작)와 이름
CLASS_NAMES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}
# 기본 고정 신뢰도. 임계값 스윕은 --conf 인자로 바꿔 여러 번 실행한다.
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """비율을 반환하되 분모가 0인 정의 불가 상황은 null로 유지한다."""

    return float(numerator / denominator) if denominator else None


def binary_scores(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """단일 클래스의 픽셀 단위 진단 지표(evaluate_test_seg.py와 동일 정의)."""

    intersection = int(np.count_nonzero(ground_truth & prediction))
    union = int(np.count_nonzero(ground_truth | prediction))
    gt_pixels = int(np.count_nonzero(ground_truth))
    pred_pixels = int(np.count_nonzero(prediction))
    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "iou": safe_ratio(intersection, union),
        "pixel_recall": safe_ratio(intersection, gt_pixels),
        "miss_ratio": safe_ratio(gt_pixels - intersection, gt_pixels),
    }


def load_coco_ground_truth(valid_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """COCO valid 어노테이션을 읽어 이미지 목록과 이미지별 어노테이션을 반환한다."""

    document = json.loads((valid_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    categories = {entry["id"]: entry["name"] for entry in document["categories"]}
    expected = {class_id + 1: name for class_id, name in CLASS_NAMES.items()}
    if categories != expected:
        raise ValueError(f"COCO 카테고리가 실험 정책과 다릅니다: {categories!r}")
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in document["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    return document["images"], by_image


def rasterize_ground_truth(
    info: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[int, np.ndarray]:
    """GT 폴리곤을 클래스별 합집합 이진 마스크로 래스터화한다."""

    import cv2

    height, width = int(info["height"]), int(info["width"])
    masks = {class_id: np.zeros((height, width), dtype=bool) for class_id in CLASS_NAMES}
    for annotation in annotations:
        class_id = int(annotation["category_id"]) - 1
        for polygon in annotation["segmentation"]:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            canvas = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
            masks[class_id] |= canvas.astype(bool)
    return masks


class UltralyticsBackend:
    """실험 H(YOLO-seg) 추론 백엔드."""

    def __init__(self, weights: Path, confidence: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.confidence = confidence

    def predict_union_masks(self, image_path: Path, height: int, width: int) -> dict[int, np.ndarray]:
        import cv2

        result = self.model.predict(
            str(image_path), conf=self.confidence, iou=0.7, verbose=False
        )[0]
        masks = {class_id: np.zeros((height, width), dtype=bool) for class_id in CLASS_NAMES}
        if result.masks is None or result.boxes is None:
            return masks
        mask_data = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for mask, class_id in zip(mask_data, classes):
            if class_id not in CLASS_NAMES:
                continue
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks[class_id] |= mask > 0.5
        return masks


class RfdetrBackend:
    """실험 J(RF-DETR-Seg) 추론 백엔드."""

    def __init__(self, weights: Path, variant: str, confidence: float) -> None:
        import rfdetr

        if not hasattr(rfdetr, variant):
            raise ValueError(f"rfdetr에 없는 모델 변형입니다: {variant}")
        self.model = getattr(rfdetr, variant)(pretrain_weights=str(weights))
        self.confidence = confidence
        self.class_id_base: int | None = None

    def predict_union_masks(self, image_path: Path, height: int, width: int) -> dict[int, np.ndarray]:
        import cv2
        from PIL import Image

        detections = self.model.predict(
            Image.open(image_path).convert("RGB"), threshold=self.confidence
        )
        masks = {class_id: np.zeros((height, width), dtype=bool) for class_id in CLASS_NAMES}
        if detections.mask is None or len(detections) == 0:
            return masks
        raw_ids = np.asarray(detections.class_id, dtype=int)
        # 예측 class id가 0시작인지 1시작(COCO category)인지 첫 관측으로 확정해 기록한다.
        if self.class_id_base is None:
            self.class_id_base = 1 if raw_ids.max(initial=0) >= len(CLASS_NAMES) else 0
        for mask, raw_id in zip(np.asarray(detections.mask), raw_ids):
            class_id = int(raw_id) - self.class_id_base
            if class_id not in CLASS_NAMES:
                continue
            mask = mask.astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks[class_id] |= mask.astype(bool)
        return masks


def aggregate(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    """부분집합(custom/public)별로 기존 보고서와 같은 통계를 만든다."""

    summary: dict[str, Any] = {"subset": subset, "images": len(rows)}
    for class_id, name in CLASS_NAMES.items():
        with_gt = [row for row in rows if row[f"{name}_gt_pixels"] > 0]
        ious = [row[f"{name}_iou"] for row in with_gt if row[f"{name}_iou"] is not None]
        complete_miss = sum(1 for row in with_gt if row[f"{name}_pred_pixels"] == 0)
        heavy_miss = sum(
            1
            for row in with_gt
            if row[f"{name}_miss_ratio"] is not None and row[f"{name}_miss_ratio"] >= 0.5
        )
        summary[name] = {
            "images_with_gt": len(with_gt),
            "complete_miss": complete_miss,
            "miss_ratio_ge_50pct": heavy_miss,
            "mean_pixel_iou": round(statistics.mean(ious), 4) if ious else None,
            "median_pixel_iou": round(statistics.median(ious), 4) if ious else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ultralytics", "rfdetr"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", default="RFDETRSegNano", help="rfdetr 백엔드의 모델 변형")
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=Path("data/processed/rfdetr_mixed_grasp_type_v2/valid"),
        help="공용 GT로 쓰는 COCO valid 폴더",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="고정 신뢰도 임계값(임계값 스윕용)",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    valid_dir = resolve(args.valid_dir)
    weights = resolve(args.weights)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"기존 결과 폴더를 덮어쓰지 않습니다: {output_dir}")

    images, annotations_by_image = load_coco_ground_truth(valid_dir)
    if args.backend == "ultralytics":
        backend: Any = UltralyticsBackend(weights, args.conf)
    else:
        backend = RfdetrBackend(weights, args.variant, args.conf)

    rows: list[dict[str, Any]] = []
    for info in images:
        image_path = valid_dir / info["file_name"]
        ground_truth = rasterize_ground_truth(info, annotations_by_image.get(info["id"], []))
        prediction = backend.predict_union_masks(image_path, int(info["height"]), int(info["width"]))
        row: dict[str, Any] = {
            "image": info["file_name"],
            "subset": "custom" if info["file_name"].startswith("custom__") else "public",
        }
        for class_id, name in CLASS_NAMES.items():
            scores = binary_scores(ground_truth[class_id], prediction[class_id])
            for key, value in scores.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f"{len(rows)}/{len(images)}장 평가", flush=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(rows[0].keys())
    with (output_dir / "per_image_pixel_iou.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "confidence_threshold": args.conf,
            "ground_truth": str(valid_dir / "_annotations.coco.json"),
            "definition": "클래스별 합집합 마스크 IoU (evaluate_test_seg.py binary_scores와 동일)",
        },
        "backend": args.backend,
        "weights": str(weights),
        "subsets": {
            "custom_val_mug03": aggregate([row for row in rows if row["subset"] == "custom"], "custom"),
            "public_val": aggregate([row for row in rows if row["subset"] == "public"], "public"),
        },
    }
    if args.backend == "rfdetr":
        summary["rfdetr_class_id_base"] = backend.class_id_base
    with (output_dir / "pixel_iou_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary["subsets"], ensure_ascii=False, indent=2, sort_keys=True))
    print("픽셀 IoU 검수를 마쳤습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
