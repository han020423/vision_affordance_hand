#!/usr/bin/env python
"""파지 종류 3클래스 모델을 공개 test·자체 val·contain 정책 감사로 평가한다.

기존 `evaluate_test_seg.py`는 2클래스 전용이므로, 3클래스
(handle_grasp_region=0, body_grasp_region=1, functional_region=2) 모델은
이 스크립트로 평가한다. 결과는 실행별 폴더에 JSON으로 저장한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

GRASP_TYPE_NAMES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}
FUNCTIONAL_CLASS_ID = 2


def metric_group(group, names: dict[int, str]) -> dict[str, object]:
    """Ultralytics Metric 객체를 클래스별 수치로 직렬화한다."""

    output: dict[str, object] = {
        "overall": {
            "precision": float(group.mp),
            "recall": float(group.mr),
            "map50": float(group.map50),
            "map50_95": float(group.map),
        },
        "per_class": {},
    }
    # 평가 분할에 없는 클래스는 배열에 포함되지 않으므로, 실제 존재하는
    # 클래스 목록(ap_class_index)을 기준으로 행을 해석한다.
    present = getattr(group, "ap_class_index", None)
    present_ids = [int(value) for value in present] if present is not None else list(names)
    for row, class_id in enumerate(present_ids):
        name = names.get(class_id, f"class_{class_id}")
        output["per_class"][name] = {
            "precision": float(group.p[row]),
            "recall": float(group.r[row]),
            "map50": float(group.ap50[row]),
            "map50_95": float(group.ap[row]),
        }
    return output


def run_val(model, data_yaml: Path, split: str, output_dir: Path, name: str, device: str, imgsz: int) -> dict[str, object]:
    """지정 분할에 대해 공식 mAP 평가를 실행하고 요약을 반환한다."""

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=True,
        project=str(output_dir),
        name=name,
        exist_ok=False,
    )
    names = {int(key): str(value) for key, value in metrics.names.items()}
    return {
        "split": split,
        "data": str(data_yaml),
        "boxes": metric_group(metrics.box, names),
        "masks": metric_group(metrics.seg, names),
    }


def pixel_iou_eval(
    model, data_yaml: Path, split: str, device: str, imgsz: int, confidence: float
) -> dict[str, object]:
    """고정 신뢰도에서 클래스별 픽셀 IoU를 측정한다 (mAP 보완 지표).

    - dataset IoU(micro): 전체 이미지의 교집합/합집합 합산 — 픽셀 관점 총성능
    - mean image IoU: 정답에 해당 클래스가 있는 이미지들의 per-image IoU 평균
    """

    import cv2
    import yaml

    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in document["names"].items()}
    split_value = document.get(split)
    if split_value is None:
        raise ValueError(f"dataset.yaml에 {split} 분할이 없습니다: {data_yaml}")
    split_dir = (data_yaml.parent / str(split_value)).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"이미지 디렉터리 분할만 지원합니다: {split_dir}")
    label_dir = data_yaml.parent / "labels" / split_dir.name
    image_count = sum(1 for p in split_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    intersect = {c: 0 for c in names}
    union = {c: 0 for c in names}
    image_ious: dict[int, list[float]] = {c: [] for c in names}

    # 경로 리스트를 통째로 넘기면 Ultralytics가 모든 파일을 동시에 열어
    # 파일 핸들이 고갈되므로, 디렉터리 소스로 스트리밍한다.
    predictions = model.predict(source=str(split_dir), imgsz=imgsz,
                                conf=confidence, device=device, retina_masks=True,
                                stream=True, verbose=False)
    for result in predictions:
        image_path = Path(result.path)
        height, width = result.orig_shape
        gt = {c: np.zeros((height, width), dtype=bool) for c in names}
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 7:
                    continue
                class_id = int(fields[0])
                points = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
                points[:, 0] *= width
                points[:, 1] *= height
                canvas = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(canvas, [points.astype(np.int32)], 1)
                gt[class_id] |= canvas.astype(bool)
        pred = {c: np.zeros((height, width), dtype=bool) for c in names}
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for mask, class_id in zip(result.masks.data.cpu().numpy(), classes):
                if class_id not in pred:
                    continue
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                pred[class_id] |= mask > 0.5
        for class_id in names:
            inter = int(np.count_nonzero(gt[class_id] & pred[class_id]))
            uni = int(np.count_nonzero(gt[class_id] | pred[class_id]))
            intersect[class_id] += inter
            union[class_id] += uni
            if gt[class_id].any():
                image_ious[class_id].append(inter / uni if uni else 0.0)

    output: dict[str, object] = {"confidence": confidence, "image_count": image_count}
    dataset_values = []
    for class_id, name in names.items():
        dataset_iou = intersect[class_id] / union[class_id] if union[class_id] else None
        per_image = image_ious[class_id]
        output[name] = {
            "dataset_iou": round(dataset_iou, 4) if dataset_iou is not None else None,
            "mean_image_iou": round(float(np.mean(per_image)), 4) if per_image else None,
            "gt_present_images": len(per_image),
        }
        if dataset_iou is not None:
            dataset_values.append(dataset_iou)
    output["mean_dataset_iou"] = round(float(np.mean(dataset_values)), 4) if dataset_values else None
    return output


def policy_audit(model, manifest_path: Path, device: str, imgsz: int, confidence: float) -> dict[str, object]:
    """contain=ignore 픽셀 위의 functional 오예측을 감사한다."""

    import cv2

    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    image_paths = [str(resolve(row["image_path"])) for row in records]
    predictions = model.predict(
        source=image_paths, imgsz=imgsz, conf=confidence, device=device,
        retina_masks=True, stream=True, verbose=False,
    )
    images_with_violation = 0
    for source, result in zip(records, predictions):
        semantic = cv2.imread(str(resolve(source["semantic_mask_path"])), cv2.IMREAD_GRAYSCALE)
        height, width = semantic.shape
        functional_pred = np.zeros((height, width), dtype=bool)
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for mask, class_id in zip(result.masks.data.detach().cpu().numpy(), classes):
                if class_id != FUNCTIONAL_CLASS_ID:
                    continue
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                functional_pred |= mask > 0.5
        ignore = semantic == 255
        overlap = int(np.count_nonzero(functional_pred & ignore))
        ignore_pixels = int(np.count_nonzero(ignore))
        if overlap > 0:
            images_with_violation += 1
        rows.append(
            {
                "source_id": source["source_id"],
                "functional_on_ignore_pixels": overlap,
                "ignore_covered_ratio": overlap / ignore_pixels if ignore_pixels else None,
            }
        )
    return {
        "image_count": len(rows),
        "images_with_functional_on_contain": images_with_violation,
        "prediction_confidence": confidence,
        "per_image": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True, help="공개 3클래스 dataset.yaml")
    parser.add_argument("--custom-data", type=Path, required=True, help="자체 3클래스 dataset.yaml")
    parser.add_argument("--policy-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--policy-confidence", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 평가 디렉터리를 재사용하지 않습니다: {output_dir}")
    output_dir.mkdir(parents=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPOSITORY_ROOT / ".ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    summary: dict[str, object] = {"model": str(args.model.resolve())}
    summary["public_test"] = run_val(
        model, args.public_data.resolve(), "test", output_dir, "public_test", args.device, args.imgsz
    )
    summary["custom_val"] = run_val(
        model, args.custom_data.resolve(), "val", output_dir, "custom_val", args.device, args.imgsz
    )
    # 픽셀 IoU: mAP와 함께 항상 보고한다 (고정 conf 0.25).
    summary["public_test"]["pixel_iou"] = pixel_iou_eval(
        model, args.public_data.resolve(), "test", args.device, args.imgsz, args.policy_confidence
    )
    summary["custom_val"]["pixel_iou"] = pixel_iou_eval(
        model, args.custom_data.resolve(), "val", args.device, args.imgsz, args.policy_confidence
    )
    if args.policy_manifest is not None:
        summary["container_policy_audit"] = policy_audit(
            model, args.policy_manifest.resolve(), args.device, args.imgsz, args.policy_confidence
        )

    (output_dir / "grasp_type_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        "public_test_mask": summary["public_test"]["masks"]["overall"],
        "public_test_iou": summary["public_test"]["pixel_iou"]["mean_dataset_iou"],
        "custom_val_mask": summary["custom_val"]["masks"]["overall"],
        "custom_val_iou": summary["custom_val"]["pixel_iou"]["mean_dataset_iou"],
    }
    if "container_policy_audit" in summary:
        compact["policy_violation_images"] = summary["container_policy_audit"][
            "images_with_functional_on_contain"
        ]
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
