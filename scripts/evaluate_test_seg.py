#!/usr/bin/env python
"""고정된 YOLO 분할 체크포인트를 평가하고 검수 가능한 오버레이를 생성한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "grasp_region", 1: "functional_region"}
CLASS_COLORS = {
    0: np.array([40, 190, 70], dtype=np.float32),
    1: np.array([210, 60, 210], dtype=np.float32),
}
IGNORE_COLOR = np.array([30, 210, 240], dtype=np.float32)

# 클래스 이름과 출력 JSON 키는 기존 학습·평가 도구와의 호환성을 위해 영문을 유지한다.
# 사람이 읽는 설명, 예외 메시지와 보고서는 한국어로 작성한다.


def sha256_file(path: Path) -> str:
    """감사 기록에 사용할 대문자 SHA-256 해시를 반환한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    """데이터 분할 정의를 바꾸지 않고 YOLO 데이터셋 YAML을 읽는다."""

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if document.get("names") != CLASS_NAMES:
        raise ValueError(f"예상하지 못한 클래스 매핑입니다({path}): {document.get('names')!r}")
    if "test" not in document:
        raise ValueError(f"데이터셋 YAML에 test 분할이 없습니다: {path}")
    return document


def resolve_split_path(dataset_yaml: Path, value: str) -> Path:
    """YAML 파일 위치를 기준으로 데이터 분할 경로를 해석한다."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = dataset_yaml.parent / candidate
    return candidate.resolve()


def list_images(directory: Path) -> list[Path]:
    """지원하는 이미지 파일을 항상 같은 순서로 나열한다."""

    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)


def polygons_to_masks(label_path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    """YOLO 분할 폴리곤을 클래스별 이진 마스크 하나로 래스터화한다."""

    masks = {class_id: np.zeros((height, width), dtype=bool) for class_id in CLASS_NAMES}
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO 라벨 파일이 없습니다: {label_path}")
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"알 수 없는 클래스 {class_id}입니다: {label_path}:{line_number}")
        coordinates = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
        if coordinates.size < 6 or coordinates.size % 2:
            raise ValueError(f"잘못된 폴리곤입니다: {label_path}:{line_number}")
        points = coordinates.reshape(-1, 2)
        points[:, 0] = np.clip(np.rint(points[:, 0] * width), 0, width - 1)
        points[:, 1] = np.clip(np.rint(points[:, 1] * height), 0, height - 1)
        polygon = points.astype(np.int32)
        temporary = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(temporary, [polygon], 1)
        masks[class_id] |= temporary.astype(bool)
    return masks


def prediction_masks(result: Any, width: int, height: int) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """예측 인스턴스를 원본 이미지 크기에서 의미 클래스별로 합친다."""

    masks = {class_id: np.zeros((height, width), dtype=bool) for class_id in CLASS_NAMES}
    max_confidence = {class_id: 0.0 for class_id in CLASS_NAMES}
    if result.masks is None or result.boxes is None:
        return masks, max_confidence
    mask_data = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    for mask, class_id, confidence in zip(mask_data, classes, confidences):
        if class_id not in CLASS_NAMES:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[class_id] |= mask > 0.5
        max_confidence[class_id] = max(max_confidence[class_id], float(confidence))
    return masks, max_confidence


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """비율을 반환하되 분모가 0인 정의 불가 상황은 null로 유지한다."""

    return float(numerator / denominator) if denominator else None


def binary_scores(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    """단일 클래스의 픽셀 단위 진단 지표를 계산 근거가 드러나게 구한다."""

    intersection = int(np.count_nonzero(ground_truth & prediction))
    union = int(np.count_nonzero(ground_truth | prediction))
    gt_pixels = int(np.count_nonzero(ground_truth))
    pred_pixels = int(np.count_nonzero(prediction))
    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "intersection_pixels": intersection,
        "iou": safe_ratio(intersection, union),
        "dice": safe_ratio(2 * intersection, gt_pixels + pred_pixels),
        "pixel_precision": safe_ratio(intersection, pred_pixels),
        "pixel_recall": safe_ratio(intersection, gt_pixels),
        "miss_ratio": safe_ratio(gt_pixels - intersection, gt_pixels),
    }


def apply_overlay(image: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """프로젝트 클래스 색을 이미지에 겹치고 마스크 경계를 그린다."""

    output = image.astype(np.float32).copy()
    for class_id, mask in masks.items():
        output[mask] = output[mask] * (1.0 - alpha) + CLASS_COLORS[class_id] * alpha
    rendered = np.clip(output, 0, 255).astype(np.uint8)
    for class_id, mask in masks.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(value) for value in CLASS_COLORS[class_id])
        cv2.drawContours(rendered, contours, -1, color, 2)
    return rendered


def apply_policy_overlay(image: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    """파지·기능·contain 무시 픽셀을 재매핑하지 않고 그대로 표시한다."""

    output = image.astype(np.float32).copy()
    colors = {1: CLASS_COLORS[0], 2: CLASS_COLORS[1], 255: IGNORE_COLOR}
    for value, color in colors.items():
        selected = semantic_mask == value
        output[selected] = output[selected] * 0.55 + color * 0.45
    return np.clip(output, 0, 255).astype(np.uint8)


def titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    """이미지 패널 위에 간결한 제목 띠를 추가한다."""

    strip = np.full((34, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(strip, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([strip, image])


def save_comparison(
    path: Path,
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    policy_mode: bool = False,
) -> None:
    """원본·정답/정책·예측 패널을 나란히 저장한다."""

    if policy_mode:
        reference = apply_policy_overlay(image, ground_truth)
        reference_title = "POLICY GT: green=grasp, yellow=ignore(contain)"
    else:
        reference = apply_overlay(image, {0: ground_truth == 1, 1: ground_truth == 2})
        reference_title = "GROUND TRUTH"
    predicted = apply_overlay(image, {0: prediction == 1, 1: prediction == 2})
    panels = [
        titled_panel(image, "ORIGINAL"),
        titled_panel(reference, reference_title),
        titled_panel(predicted, "PREDICTION: green=grasp, magenta=functional"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"오버레이를 저장하지 못했습니다: {path}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """모든 행의 키 합집합을 일정한 순서로 유지해 딕셔너리를 저장한다."""

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def optional_float(value: Any) -> float | None:
    """사용 불가 값은 그대로 보존하면서 감사 지표 숫자를 해석한다."""

    if value in (None, ""):
        return None
    return float(value)


def summarize_formal_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """AP 지표와 섞지 않고 고정 신뢰도 픽셀 진단 결과를 요약한다."""

    class_summary: dict[str, Any] = {}
    for class_name, prefix in (("grasp_region", "grasp"), ("functional_region", "functional")):
        present = [row for row in rows if int(float(row[f"{prefix}_gt_pixels"])) > 0]
        ious = [
            value
            for row in present
            if (value := optional_float(row[f"{prefix}_iou"])) is not None
        ]
        class_summary[class_name] = {
            "ground_truth_present_images": len(present),
            "prediction_absent_images": sum(
                int(float(row[f"{prefix}_pred_pixels"])) == 0 for row in present
            ),
            "miss_ratio_at_least_0_5_images": sum(
                (optional_float(row[f"{prefix}_miss_ratio"]) or 0.0) >= 0.5 for row in present
            ),
            "mean_pixel_iou": statistics.mean(ious) if ious else None,
            "median_pixel_iou": statistics.median(ious) if ious else None,
        }

    object_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        object_rows.setdefault(str(row["object_id"]), []).append(row)
    by_object: dict[str, Any] = {}
    for object_id, samples in sorted(object_rows.items()):
        object_summary: dict[str, Any] = {"image_count": len(samples)}
        for class_name, prefix in (("grasp_region", "grasp"), ("functional_region", "functional")):
            present = [row for row in samples if int(float(row[f"{prefix}_gt_pixels"])) > 0]
            ious = [
                value
                for row in present
                if (value := optional_float(row[f"{prefix}_iou"])) is not None
            ]
            object_summary[class_name] = {
                "ground_truth_present_images": len(present),
                "prediction_absent_images": sum(
                    int(float(row[f"{prefix}_pred_pixels"])) == 0 for row in present
                ),
                "mean_pixel_iou": statistics.mean(ious) if ious else None,
            }
        by_object[object_id] = object_summary

    def overlap_count(key: str, threshold: float) -> int:
        return sum(
            value > threshold
            for row in rows
            if (value := optional_float(row[key])) is not None
        )

    ranked = sorted(
        rows,
        key=lambda row: optional_float(row["review_error_score"]) or -1.0,
        reverse=True,
    )
    return {
        "evaluation_kind": "fixed_confidence_pixel_review_separate_from_formal_ap_metrics",
        "image_count": len(rows),
        "classes": class_summary,
        "class_overlap_review": {
            "note": (
                "이 비율은 예측 클래스 마스크 중 다른 정답 클래스와 겹친 부분의 비중입니다. "
                "AP 지표가 아니라 육안 검수 우선순위를 정하는 표시값입니다."
            ),
            "predicted_functional_overlapping_gt_grasp_over_0_1_images": overlap_count(
                "pred_functional_on_gt_grasp_ratio", 0.1
            ),
            "predicted_functional_overlapping_gt_grasp_over_0_5_images": overlap_count(
                "pred_functional_on_gt_grasp_ratio", 0.5
            ),
            "predicted_grasp_overlapping_gt_functional_over_0_1_images": overlap_count(
                "pred_grasp_on_gt_functional_ratio", 0.1
            ),
            "predicted_grasp_overlapping_gt_functional_over_0_5_images": overlap_count(
                "pred_grasp_on_gt_functional_ratio", 0.5
            ),
        },
        "by_object": by_object,
        "worst_images": [
            {
                "image": row["image"],
                "object_id": row["object_id"],
                "review_error_score": optional_float(row["review_error_score"]),
                "grasp_iou": optional_float(row["grasp_iou"]),
                "functional_iou": optional_float(row["functional_iou"]),
                "overlay_path": row["overlay_path"],
            }
            for row in ranked[:12]
        ],
    }


def make_contact_sheet(
    destination: Path,
    ranked_rows: Sequence[dict[str, Any]],
    overlay_key: str,
    caption_keys: Iterable[str],
    limit: int = 10,
) -> None:
    """이미 렌더링한 비교 이미지로 간결한 검수표를 만든다."""

    tiles: list[np.ndarray] = []
    for row in ranked_rows[:limit]:
        image = cv2.imread(str(row[overlay_key]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_width = 960
        target_height = max(1, int(image.shape[0] * target_width / image.shape[1]))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        caption = " | ".join(f"{key}={row.get(key)}" for key in caption_keys)
        tiles.append(titled_panel(image, caption[:150]))
    if not tiles:
        return
    width = max(tile.shape[1] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, 0, 0, width - tile.shape[1], cv2.BORDER_CONSTANT, value=255) for tile in tiles]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 92])


def metric_group(group: Any, names: dict[int, str]) -> dict[str, Any]:
    """콘솔 문자열에 의존하지 않고 Ultralytics Metric 객체를 직렬화한다."""

    output: dict[str, Any] = {
        "overall": {
            "precision": float(group.mp),
            "recall": float(group.mr),
            "map50": float(group.map50),
            "map50_95": float(group.map),
        },
        "per_class": {},
    }
    for class_id, name in names.items():
        output["per_class"][name] = {
            "precision": float(group.p[class_id]),
            "recall": float(group.r[class_id]),
            "map50": float(group.ap50[class_id]),
            "map50_95": float(group.ap[class_id]),
        }
    return output


def evaluate_formal_test(
    model: Any,
    model_path: Path,
    dataset_yaml: Path,
    test_images: Sequence[Path],
    output_dir: Path,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
    prediction_confidence: float,
) -> dict[str, Any]:
    """공식 test 지표와 별도의 고정 임계값 육안 감사를 실행한다."""

    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
        project=str(output_dir),
        name="formal_metrics",
        exist_ok=False,
    )
    names = {int(key): str(value) for key, value in metrics.names.items()}
    # 공식 mAP 계산과 고정 임계값 육안 검수는 목적이 다르므로 결과를 분리해 저장한다.
    formal_metrics = {
        "evaluation_kind": "formal_held_out_test",
        "split": "test",
        "image_count": len(test_images),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ultralytics_results_dict": {key: float(value) for key, value in metrics.results_dict.items()},
        "boxes": metric_group(metrics.box, names),
        "masks": metric_group(metrics.seg, names),
        "note": "이 파일은 test 분할 결과만 포함하며 학습 중 validation 지표와 분리되어 있습니다.",
    }
    (output_dir / "formal_test_metrics.json").write_text(
        json.dumps(formal_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overlays_dir = output_dir / "formal_test_overlays"
    labels_dir = dataset_yaml.parent / "labels" / "test"
    rows: list[dict[str, Any]] = []
    predictions = model.predict(
        source=[str(path) for path in test_images],
        imgsz=imgsz,
        conf=prediction_confidence,
        iou=0.7,
        device=device,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    for result in predictions:
        image_path = Path(result.path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"test 이미지를 읽지 못했습니다: {image_path}")
        height, width = image.shape[:2]
        gt_masks = polygons_to_masks(labels_dir / f"{image_path.stem}.txt", width, height)
        pred_masks, confidences = prediction_masks(result, width, height)
        # 표시용 의미값 1과 2를 사용해 원본 라벨 ID와 시각화 단계를 명확히 구분한다.
        gt_semantic = np.zeros((height, width), dtype=np.uint8)
        pred_semantic = np.zeros((height, width), dtype=np.uint8)
        gt_semantic[gt_masks[0]] = 1
        gt_semantic[gt_masks[1]] = 2
        pred_semantic[pred_masks[0]] = 1
        pred_semantic[pred_masks[1]] = 2
        overlay_path = overlays_dir / f"{image_path.stem}.jpg"
        save_comparison(overlay_path, image, gt_semantic, pred_semantic)

        grasp = binary_scores(gt_masks[0], pred_masks[0])
        functional = binary_scores(gt_masks[1], pred_masks[1])
        pred_union = pred_masks[0] | pred_masks[1]
        gt_union = gt_masks[0] | gt_masks[1]
        valid_ious = [score for score in (grasp["iou"], functional["iou"]) if score is not None]
        mean_iou = float(np.mean(valid_ious)) if valid_ious else None
        row: dict[str, Any] = {
            "image": image_path.name,
            "object_id": "__".join(image_path.stem.split("__")[:2]),
            "overlay_path": str(overlay_path),
            "mean_pixel_iou": mean_iou,
            "grasp_confidence_max": confidences[0],
            "functional_confidence_max": confidences[1],
            "pred_functional_on_gt_grasp_ratio": safe_ratio(
                int(np.count_nonzero(pred_masks[1] & gt_masks[0])), int(np.count_nonzero(pred_masks[1]))
            ),
            "pred_grasp_on_gt_functional_ratio": safe_ratio(
                int(np.count_nonzero(pred_masks[0] & gt_masks[1])), int(np.count_nonzero(pred_masks[0]))
            ),
            "prediction_on_background_ratio": safe_ratio(
                int(np.count_nonzero(pred_union & ~gt_union)), int(np.count_nonzero(pred_union))
            ),
        }
        row.update({f"grasp_{key}": value for key, value in grasp.items()})
        row.update({f"functional_{key}": value for key, value in functional.items()})
        row["review_error_score"] = None if mean_iou is None else 1.0 - mean_iou
        rows.append(row)

    write_csv(output_dir / "formal_test_per_image_audit.csv", rows)
    (output_dir / "formal_test_review_summary.json").write_text(
        json.dumps(summarize_formal_rows(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ranked = sorted(rows, key=lambda row: float(row["review_error_score"] or -1.0), reverse=True)
    make_contact_sheet(
        output_dir / "formal_review_sheets" / "worst_overall.jpg",
        ranked,
        "overlay_path",
        ("image", "mean_pixel_iou", "grasp_miss_ratio", "functional_miss_ratio"),
    )
    grasp_missing = sorted(rows, key=lambda row: float(row["grasp_miss_ratio"] or -1.0), reverse=True)
    make_contact_sheet(
        output_dir / "formal_review_sheets" / "worst_grasp_missing.jpg",
        grasp_missing,
        "overlay_path",
        ("image", "grasp_iou", "grasp_miss_ratio"),
    )
    functional_missing = sorted(
        rows, key=lambda row: float(row["functional_miss_ratio"] or -1.0), reverse=True
    )
    make_contact_sheet(
        output_dir / "formal_review_sheets" / "worst_functional_missing.jpg",
        functional_missing,
        "overlay_path",
        ("image", "functional_iou", "functional_miss_ratio"),
    )
    return formal_metrics


def load_policy_manifest(path: Path) -> list[dict[str, Any]]:
    """공식 평가와 명시적으로 분리된 용기 정책 감사 manifest를 읽는다."""

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        for key in ("object_id", "source_id", "image_path", "semantic_mask_path"):
            if key not in row:
                raise ValueError(f"정책 manifest의 {line_number}번째 줄에 {key}가 없습니다")
        rows.append(row)
    return rows


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    """manifest 위치를 기준으로 정책 감사 자료의 경로를 해석한다."""

    candidate = Path(value)
    return candidate if candidate.is_absolute() else (manifest.parent / candidate).resolve()


def evaluate_container_policy(
    model: Any,
    manifest_path: Path,
    output_dir: Path,
    device: str,
    imgsz: int,
    confidence: float,
) -> dict[str, Any]:
    """ignore로 유지한 UMD contain 픽셀의 기능 영역 오예측을 감사한다."""

    manifest_rows = load_policy_manifest(manifest_path)
    audit_dir = output_dir / "container_policy_audit"
    overlays_dir = audit_dir / "overlays"
    image_paths = [resolve_manifest_path(manifest_path, row["image_path"]) for row in manifest_rows]
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=confidence,
        iou=0.7,
        device=device,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    rows: list[dict[str, Any]] = []
    for source, result in zip(manifest_rows, predictions):
        image_path = resolve_manifest_path(manifest_path, source["image_path"])
        semantic_path = resolve_manifest_path(manifest_path, source["semantic_mask_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        # 255인 contain 픽셀은 background로 바꾸지 않고 정책 위반 검사 대상으로 유지한다.
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        if image is None or semantic is None:
            raise OSError(f"정책 감사 이미지·마스크 쌍을 읽지 못했습니다: {image_path}, {semantic_path}")
        height, width = image.shape[:2]
        if semantic.shape != (height, width):
            raise ValueError(f"정책 감사 이미지와 마스크 크기가 다릅니다: {source['source_id']}")
        pred_masks, confidences = prediction_masks(result, width, height)
        pred_semantic = np.zeros((height, width), dtype=np.uint8)
        pred_semantic[pred_masks[0]] = 1
        pred_semantic[pred_masks[1]] = 2
        overlay_path = overlays_dir / f"{source['source_id'].replace('/', '__')}.jpg"
        save_comparison(overlay_path, image, semantic, pred_semantic, policy_mode=True)
        ignore = semantic == 255
        gt_grasp = semantic == 1
        functional_on_ignore = int(np.count_nonzero(pred_masks[1] & ignore))
        pred_functional_pixels = int(np.count_nonzero(pred_masks[1]))
        ignore_pixels = int(np.count_nonzero(ignore))
        grasp = binary_scores(gt_grasp, pred_masks[0])
        rows.append(
            {
                "object_id": source["object_id"],
                "source_id": source["source_id"],
                "overlay_path": str(overlay_path),
                "ignore_contain_pixels": ignore_pixels,
                "pred_functional_pixels": pred_functional_pixels,
                "functional_on_ignore_pixels": functional_on_ignore,
                "ignore_covered_by_functional_ratio": safe_ratio(functional_on_ignore, ignore_pixels),
                "pred_functional_inside_ignore_ratio": safe_ratio(
                    functional_on_ignore, pred_functional_pixels
                ),
                "functional_confidence_max": confidences[1],
                "grasp_iou": grasp["iou"],
                "grasp_miss_ratio": grasp["miss_ratio"],
            }
        )

    write_csv(audit_dir / "container_policy_per_image.csv", rows)
    ranked = sorted(
        rows,
        key=lambda row: float(row["ignore_covered_by_functional_ratio"] or 0.0),
        reverse=True,
    )
    make_contact_sheet(
        audit_dir / "worst_functional_on_contain.jpg",
        ranked,
        "overlay_path",
        ("source_id", "ignore_covered_by_functional_ratio", "functional_confidence_max"),
    )
    by_object: dict[str, dict[str, Any]] = {}
    for object_id in sorted({str(row["object_id"]) for row in rows}):
        selected = [row for row in rows if row["object_id"] == object_id]
        ratios = [float(row["ignore_covered_by_functional_ratio"] or 0.0) for row in selected]
        by_object[object_id] = {
            "image_count": len(selected),
            "images_with_any_functional_on_ignore": sum(
                int(row["functional_on_ignore_pixels"] > 0) for row in selected
            ),
            "mean_ignore_covered_by_functional_ratio": float(np.mean(ratios)),
            "max_ignore_covered_by_functional_ratio": float(np.max(ratios)),
        }
    summary = {
        "evaluation_kind": "separate_container_policy_audit_not_formal_test_metrics",
        "policy": "UMD contain은 ignore로 유지하며 컵·용기 몸통을 functional_region으로 바꾸지 않습니다.",
        "prediction_confidence": confidence,
        "image_count": len(rows),
        "by_object": by_object,
    }
    (audit_dir / "container_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prediction-confidence", type=float, default=0.25)
    parser.add_argument("--expected-test-images", type=int, default=494)
    args = parser.parse_args()

    model_path = args.model.resolve()
    dataset_yaml = args.data.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {model_path}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"데이터셋 YAML 파일이 없습니다: {dataset_yaml}")
    if output_dir.exists():
        raise FileExistsError(f"기존 평가 디렉터리를 재사용하지 않습니다: {output_dir}")

    document = load_dataset_yaml(dataset_yaml)
    test_dir = resolve_split_path(dataset_yaml, str(document["test"]))
    test_images = list_images(test_dir)
    if len(test_images) != args.expected_test_images:
        raise ValueError(
            f"test 이미지가 {args.expected_test_images}장이어야 하지만 {test_dir}에서 {len(test_images)}장을 찾았습니다"
        )
    output_dir.mkdir(parents=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPOSITORY_ROOT / ".ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    formal = evaluate_formal_test(
        model,
        model_path,
        dataset_yaml,
        test_images,
        output_dir,
        args.device,
        args.imgsz,
        args.batch,
        args.workers,
        args.prediction_confidence,
    )
    policy = None
    if args.policy_manifest is not None:
        policy = evaluate_container_policy(
            model,
            args.policy_manifest.resolve(),
            output_dir,
            args.device,
            args.imgsz,
            args.prediction_confidence,
        )
    status = {
        "status": "completed",
        "formal_test_images": formal["image_count"],
        "container_policy_audit_images": policy["image_count"] if policy else 0,
    }
    (output_dir / "evaluation_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
