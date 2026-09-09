"""통합 manifest에서 결정적인 YOLO 인스턴스 분할 데이터를 내보낸다."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from src.labeling.policy import IGNORE_VALUE, MODEL_CLASS_IDS


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_MAP = {"pretrain": "train", "train": "train", "validation": "val", "test": "test"}
IGNORE_EXPORT_POLICIES = {
    "exclude_sample",
    "contain_as_background_keep_grasp",
}


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, object]]:
    """JSONL을 읽고 잘못된 행의 정확한 위치를 함께 보고한다."""

    records: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"잘못된 JSON입니다: {path}:{line_number}") from exc
    return records


def sha256_file(path: Path) -> str:
    """데이터 출처 기록용 대문자 SHA-256 해시를 반환한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_stem(source_dataset: str, source_id: str) -> str:
    """원본 식별자로부터 충돌 가능성이 낮고 이식 가능한 파일명을 만든다."""

    readable = re.sub(r"[^A-Za-z0-9._-]+", "__", f"{source_dataset}__{source_id}").strip("._-")
    suffix = hashlib.sha256(f"{source_dataset}\0{source_id}".encode()).hexdigest()[:12]
    return f"{readable}__{suffix}"


def _bridge_ring_polygon(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """바깥 윤곽과 구멍 윤곽들을 절개선(slit)으로 연결해 단일 폴리곤을 만든다.

    한붓그리기 방식: 바깥 윤곽을 따라가다 구멍과 가장 가까운 꼭짓점에서
    구멍 윤곽으로 들어가 한 바퀴 돈 뒤 같은 지점으로 복귀한다. OpenCV의
    RETR_CCOMP는 바깥과 구멍 윤곽을 서로 반대 방향으로 반환하므로, 이렇게
    이은 단일 폴리곤을 채우면 구멍 내부는 칠해지지 않는다. 결과 품질은
    호출부에서 재구성 IoU로 검증한다.
    """

    polygon = outer.reshape(-1, 2)
    for hole in sorted(holes, key=cv2.contourArea, reverse=True):
        hole_points = hole.reshape(-1, 2)
        distances = ((polygon[:, None, :].astype(np.int64) - hole_points[None, :, :]) ** 2).sum(axis=2)
        outer_index, hole_index = np.unravel_index(np.argmin(distances), distances.shape)
        polygon = np.concatenate(
            [
                polygon[: outer_index + 1],
                hole_points[hole_index:],
                hole_points[: hole_index + 1],
                polygon[outer_index:],
            ]
        )
    return polygon


def component_to_yolo_polygon(
    instance_mask: np.ndarray,
    component: dict[str, object],
    *,
    max_secondary_contour_area: float = 0.0,
    max_hole_contour_area: float = 0.0,
    allow_ring_slit: bool = False,
    ring_slit_min_iou: float = 0.85,
) -> tuple[str | None, dict[str, object]]:
    """연결 성분 하나를 YOLO 폴리곤으로 변환한다.

    기본값은 원본 형상을 전혀 바꾸지 않는 엄격 모드다. 임계값을 지정하면
    YOLO가 표현할 수 없는 아주 작은 분리 조각과 구멍만 변환본에서 정리하고,
    변경량을 감사 정보에 남긴다. 가장 큰 전경 윤곽은 항상 보존한다.

    allow_ring_slit=True이면 컵 몸통처럼 큰 구멍을 두른 고리형 성분을
    절개선 방식 단일 폴리곤으로 표현한다. 구멍은 채우지 않고 배경으로
    보존되며, 재구성 IoU가 ring_slit_min_iou 미만이면 안전하게 제외한다.
    """

    if max_secondary_contour_area < 0 or max_hole_contour_area < 0:
        raise ValueError("YOLO 위상 정리 임계값은 0 이상이어야 합니다")

    component_id = int(component["component_id"])
    class_name = str(component["class_name"])
    expected_class_id = MODEL_CLASS_IDS.get(class_name)
    audit: dict[str, object] = {
        "component_id": component_id,
        "class_name": class_name,
        "model_class_id": expected_class_id,
        "status": "excluded",
        "reasons": [],
    }
    if expected_class_id is None or int(component.get("model_class_id", -1)) != expected_class_id:
        audit["reasons"] = ["invalid_component_class_mapping"]
        return None, audit

    source_binary = (instance_mask == component_id).astype(np.uint8)
    pixel_count = int(source_binary.sum())
    audit["pixel_count"] = pixel_count
    if pixel_count == 0:
        audit["reasons"] = ["component_id_missing_from_instance_mask"]
        return None, audit

    binary = source_binary.copy()
    cleanup_contours, cleanup_hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    removed_pixels = 0
    filled_pixels = 0
    if cleanup_hierarchy is not None:
        roots = [
            index
            for index in range(len(cleanup_contours))
            if int(cleanup_hierarchy[0][index][3]) == -1
        ]
        largest_root = (
            max(roots, key=lambda index: cv2.contourArea(cleanup_contours[index]))
            if roots
            else None
        )
        # 구멍을 먼저 채운 뒤 고립 조각을 제거한다. 고립 조각 안에 작은
        # 구멍이 있을 때 반대 순서로 처리하면 지운 조각의 일부가 되살아난다.
        for index, contour in enumerate(cleanup_contours):
            area = float(cv2.contourArea(contour))
            parent = int(cleanup_hierarchy[0][index][3])
            before = int(binary.sum())
            if parent != -1 and 0.0 < area <= max_hole_contour_area:
                cv2.drawContours(binary, [contour], -1, 1, thickness=cv2.FILLED)
                filled_pixels += int(binary.sum()) - before
        for index, contour in enumerate(cleanup_contours):
            area = float(cv2.contourArea(contour))
            parent = int(cleanup_hierarchy[0][index][3])
            if (
                parent == -1
                and index != largest_root
                and 0.0 < area <= max_secondary_contour_area
            ):
                before = int(binary.sum())
                cv2.drawContours(binary, [contour], -1, 0, thickness=cv2.FILLED)
                removed_pixels += before - int(binary.sum())
    audit["topology_cleanup"] = {
        "applied": bool(removed_pixels or filled_pixels),
        "max_secondary_contour_area": max_secondary_contour_area,
        "max_hole_contour_area": max_hole_contour_area,
        "removed_pixels": removed_pixels,
        "filled_pixels": filled_pixels,
    }
    audit["export_pixel_count"] = int(binary.sum())

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        audit["reasons"] = ["component_not_single_hole_free_polygon"]
        return None, audit
    hierarchy_rows = hierarchy[0]
    root_indices = [
        index for index in range(len(contours)) if int(hierarchy_rows[index][3]) == -1
    ]
    hole_indices = [
        index for index in range(len(contours)) if int(hierarchy_rows[index][3]) != -1
    ]
    if len(contours) == 1 and not hole_indices:
        points = contours[0].reshape(-1, 2)
    elif allow_ring_slit and len(root_indices) == 1 and hole_indices:
        # 고리형: 구멍을 채우지 않고 절개선 단일 폴리곤으로 표현한다.
        points = _bridge_ring_polygon(
            contours[root_indices[0]], [contours[index] for index in hole_indices]
        )
        audit["ring_slit"] = {"applied": True, "holes_bridged": len(hole_indices)}
    else:
        audit["reasons"] = ["component_not_single_hole_free_polygon"]
        return None, audit

    if len(points) < 3:
        audit["reasons"] = ["component_has_fewer_than_three_polygon_points"]
        return None, audit

    height, width = binary.shape
    normalized: list[float] = []
    for x, y in points:
        normalized.extend((float(x) / width, float(y) / height))
    if not all(0.0 <= value <= 1.0 for value in normalized):
        audit["reasons"] = ["normalized_polygon_coordinate_out_of_range"]
        return None, audit

    reconstructed = np.zeros_like(binary)
    cv2.fillPoly(reconstructed, [points.astype(np.int32)], 1)
    intersection = int(np.logical_and(binary, reconstructed).sum())
    union = int(np.logical_or(binary, reconstructed).sum())
    polygon_iou = intersection / union if union else 0.0
    source_intersection = int(np.logical_and(source_binary, reconstructed).sum())
    source_union = int(np.logical_or(source_binary, reconstructed).sum())
    source_polygon_iou = source_intersection / source_union if source_union else 0.0
    if audit.get("ring_slit") and polygon_iou < ring_slit_min_iou:
        # 절개선 재구성이 원본 고리를 충분히 보존하지 못하면 내보내지 않는다.
        audit["reasons"] = ["ring_slit_reconstruction_iou_too_low"]
        audit["polygon_iou"] = polygon_iou
        return None, audit
    audit.update(
        {
            "status": "exported",
            "reasons": [],
            "polygon_points": int(len(points)),
            "polygon_iou": polygon_iou,
            "source_polygon_iou": source_polygon_iou,
        }
    )
    coordinates = " ".join(f"{value:.8f}" for value in normalized)
    return f"{expected_class_id} {coordinates}", audit


def _resolve_repository_path(repository_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# contain을 ignore로 저장하는 데이터셋과, 그때 기록되는 사유 문자열.
# 이 사유 하나만 있는 샘플에만 contain 배경 정책을 적용할 수 있다.
CONTAIN_POLICY_REASONS: dict[str, str] = {
    "umd": "umd_contain_policy",
    "iit_aff": "iit_contain_policy",
}


def can_apply_contain_background_override(source: dict[str, object]) -> bool:
    """contain만 ignore인 변환 완료 샘플인지 보수적으로 판정한다.

    이 판정은 통합 라벨을 바꾸지 않는다. YOLO 폴리곤 형식에서만 contain
    폴리곤을 쓰지 않고, 이미 만들어진 grasp 컴포넌트를 학습에 사용하는
    실험을 안전하게 제한하기 위한 조건이다. UMD와 IIT-AFF처럼 contain을
    ignore로 저장하는 데이터셋에만 적용된다.
    """

    reasons = {str(value) for value in source.get("reasons", [])}
    original_labels = {str(value) for value in source.get("original_labels", [])}
    mapped_labels = {str(value) for value in source.get("mapped_labels", [])}
    expected_reason = CONTAIN_POLICY_REASONS.get(str(source.get("source_dataset", "")))
    return (
        expected_reason is not None
        and source.get("conversion_status") == "converted"
        and reasons == {expected_reason}
        and "contain" in original_labels
        and "ignore" in mapped_labels
    )


def is_verified_negative_sample(
    source: dict[str, object], semantic_mask: np.ndarray, instance_mask: np.ndarray
) -> bool:
    """사람이 확인한 순수 배경 샘플인지 보수적으로 판정한다.

    객체가 없다는 메타데이터뿐 아니라 두 마스크가 실제로 모두 0인지도
    함께 확인한다. 이 조건을 통과한 경우에만 YOLO 빈 라벨 파일을 만든다.
    """

    mapped_labels = {str(value) for value in source.get("mapped_labels", [])}
    components = list(source.get("components", []))
    return (
        source.get("is_negative") is True
        and source.get("annotation_status") == "verified_empty"
        and mapped_labels == {"background"}
        and not components
        and not np.any(semantic_mask)
        and not np.any(instance_mask)
    )


def export_public_yolo(
    manifest_paths: list[Path],
    output_root: Path,
    repository_root: Path,
    *,
    affgrasp_repeat: int = 2,
    ignore_export_policy: str = "exclude_sample",
    max_secondary_contour_area: float = 0.0,
    max_hole_contour_area: float = 0.0,
    allow_ring_slit: bool = False,
    dry_run: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """정책을 지키는 YOLO 폴리곤을 내보내고 모든 제외 판단을 기록한다."""

    if affgrasp_repeat < 1:
        raise ValueError("affgrasp_repeat는 1 이상이어야 합니다")
    if ignore_export_policy not in IGNORE_EXPORT_POLICIES:
        raise ValueError(
            "지원하지 않는 ignore 내보내기 정책입니다: "
            f"{ignore_export_policy}. 허용값={sorted(IGNORE_EXPORT_POLICIES)}"
        )
    records = read_jsonl(manifest_paths)
    export_records: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    class_instances: Counter[str] = Counter()
    split_images: Counter[str] = Counter()
    staging = output_root.with_name(output_root.name + ".tmp")
    if not dry_run:
        if output_root.exists():
            raise FileExistsError(f"출력 경로가 이미 있습니다. 다른 경로를 선택하세요: {output_root}")
        if staging.exists():
            raise FileExistsError(f"임시 출력 경로가 이미 있습니다: {staging}")
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)

    try:
        for source in records:
            source_dataset = str(source.get("source_dataset", ""))
            source_id = str(source.get("source_id", ""))
            output_split = SPLIT_MAP.get(str(source.get("split", "")))
            audit: dict[str, object] = {
                "source_dataset": source_dataset,
                "source_id": source_id,
                "object_id": str(source.get("object_id", "")),
                "source_split": str(source.get("split", "")),
                "output_split": output_split,
                "source_conversion_status": str(source.get("conversion_status", "")),
                "export_status": "excluded",
                "reasons": [],
                "component_audit": [],
                "exports": [],
                "source_policy": {
                    "original_labels": list(source.get("original_labels", [])),
                    "mapped_labels": list(source.get("mapped_labels", [])),
                    "reasons": list(source.get("reasons", [])),
                },
                "yolo_training_override": {
                    "policy": ignore_export_policy,
                    "status": "not_applied",
                    "reason": None,
                },
            }
            reasons: list[str] = []
            if source.get("conversion_status") != "converted":
                reasons.append("source_not_converted")
            if output_split is None:
                reasons.append("split_not_exportable")

            if reasons:
                audit["reasons"] = sorted(set(reasons))
                counters["excluded"] += 1
                for reason in audit["reasons"]:
                    counters[f"reason:{reason}"] += 1
                export_records.append(audit)
                continue

            image_path = _resolve_repository_path(repository_root, source.get("image_path"))
            semantic_path = _resolve_repository_path(repository_root, source.get("semantic_mask_path"))
            instance_path = _resolve_repository_path(repository_root, source.get("instance_mask_path"))
            for name, path in (
                ("image", image_path),
                ("semantic_mask", semantic_path),
                ("instance_mask", instance_path),
            ):
                if not path.is_file():
                    reasons.append(f"missing_{name}")
            if reasons:
                audit["reasons"] = sorted(set(reasons))
                counters["excluded"] += 1
                for reason in audit["reasons"]:
                    counters[f"reason:{reason}"] += 1
                export_records.append(audit)
                continue

            try:
                with Image.open(image_path) as image:
                    image_size = image.size
                semantic_mask = np.asarray(Image.open(semantic_path).convert("L"))
                instance_mask = np.asarray(Image.open(instance_path))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                audit["reasons"] = [f"unreadable_export_input:{type(exc).__name__}"]
                counters["excluded"] += 1
                counters[f"reason:{audit['reasons'][0]}"] += 1
                export_records.append(audit)
                continue

            if image_size != (semantic_mask.shape[1], semantic_mask.shape[0]):
                reasons.append("image_semantic_size_mismatch")
            if semantic_mask.shape != instance_mask.shape:
                reasons.append("semantic_instance_size_mismatch")
            ignore_pixel_count = int(np.sum(semantic_mask == IGNORE_VALUE))
            audit["ignore_pixel_count"] = ignore_pixel_count
            has_ignore = ignore_pixel_count > 0
            contain_override_applied = False
            if has_ignore:
                contain_override_applied = (
                    ignore_export_policy == "contain_as_background_keep_grasp"
                    and can_apply_contain_background_override(source)
                )
                if contain_override_applied:
                    # 통합 semantic PNG는 그대로 두고 YOLO 라벨에 존재하는
                    # grasp 컴포넌트만 기록한다. contain은 암묵적 배경이 된다.
                    audit["yolo_training_override"] = {
                        "policy": ignore_export_policy,
                        "status": "applied",
                        "reason": "umd_contain_ignore_treated_as_implicit_background",
                    }
                    counters["contain_background_override_applied"] += 1
                else:
                    reasons.append("ignore_pixels_not_supported_by_yolo_polygon")
            if reasons:
                audit["reasons"] = sorted(set(reasons))
                counters["excluded"] += 1
                for reason in audit["reasons"]:
                    counters[f"reason:{reason}"] += 1
                export_records.append(audit)
                continue

            lines: list[str] = []
            component_audit: list[dict[str, object]] = []
            local_class_instances: Counter[str] = Counter()
            for component in sorted(
                source.get("components", []), key=lambda value: int(value["component_id"])
            ):
                line, component_result = component_to_yolo_polygon(
                    instance_mask,
                    component,
                    max_secondary_contour_area=max_secondary_contour_area,
                    max_hole_contour_area=max_hole_contour_area,
                    allow_ring_slit=allow_ring_slit,
                )
                component_audit.append(component_result)
                if line is not None:
                    lines.append(line)
                    local_class_instances[str(component_result["class_name"])] += 1
                    if component_result.get("topology_cleanup", {}).get("applied"):
                        counters["components_with_topology_cleanup"] += 1
                    if component_result.get("ring_slit", {}).get("applied"):
                        counters["components_with_ring_slit"] += 1
            audit["component_audit"] = component_audit
            failed_components = [
                result for result in component_audit if result["status"] != "exported"
            ]
            for result in failed_components:
                counters["components_excluded"] += 1
                for reason in result.get("reasons", []):
                    counters[f"component_reason:{reason}"] += 1
            verified_negative = is_verified_negative_sample(
                source, semantic_mask, instance_mask
            )
            if not lines and not verified_negative:
                audit["reasons"] = ["no_exportable_components"]
                counters["excluded"] += 1
                counters["reason:no_exportable_components"] += 1
                export_records.append(audit)
                continue

            if verified_negative:
                # 배경 영상은 객체 라인이 없는 0바이트 txt가 정답이다.
                audit["negative_sample"] = True
                counters["verified_negative_samples_exported"] += 1

            repeats = affgrasp_repeat if source_dataset == "affgrasp" else 1
            stem = safe_stem(source_dataset, source_id)
            suffix = image_path.suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                suffix = ".jpg"
            exported_paths: list[dict[str, object]] = []
            for repeat_index in range(repeats):
                repeat_suffix = f"__r{repeat_index:02d}" if repeats > 1 else ""
                filename_stem = f"{stem}{repeat_suffix}"
                relative_image = Path("images") / str(output_split) / f"{filename_stem}{suffix}"
                relative_label = Path("labels") / str(output_split) / f"{filename_stem}.txt"
                if not dry_run:
                    target_image = staging / relative_image
                    if image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                        shutil.copy2(image_path, target_image)
                    else:
                        Image.open(image_path).convert("RGB").save(target_image)
                    label_text = "\n".join(lines) + ("\n" if lines else "")
                    (staging / relative_label).write_text(label_text, encoding="utf-8")
                exported_paths.append(
                    {
                        "repeat_index": repeat_index,
                        "image_path": relative_image.as_posix(),
                        "label_path": relative_label.as_posix(),
                    }
                )
                split_images[str(output_split)] += 1

            audit["export_status"] = (
                "exported_negative"
                if verified_negative
                else "exported_with_component_exclusions"
                if failed_components
                else "exported"
            )
            audit["reasons"] = (
                ["unrepresentable_components_excluded"] if failed_components else []
            )
            audit["exports"] = exported_paths
            audit["instance_count"] = len(lines)
            if contain_override_applied:
                audit["training_semantics_note"] = (
                    "통합 마스크의 contain=ignore는 보존되며, 이 YOLO 변환본에서만 "
                    "contain 픽셀을 암묵적 배경으로 사용합니다."
                )
            class_instances.update(local_class_instances)
            counters["exported_unique_sources"] += 1
            counters["exported_images_with_repeats"] += repeats
            export_records.append(audit)

        summary: dict[str, object] = {
            "schema_version": 1,
            "policy": {
                "classes": {0: "grasp_region", 1: "functional_region"},
                "ignore_index": IGNORE_VALUE,
                "ignore_export_strategy": ignore_export_policy,
                "source_unified_policy_unchanged": "umd_contain_maps_to_ignore_255",
                "unrepresentable_component_strategy": "exclude_component_and_log_reason",
                "topology_cleanup": {
                    "max_secondary_contour_area": max_secondary_contour_area,
                    "max_hole_contour_area": max_hole_contour_area,
                    "source_masks_modified": False,
                },
                "ring_slit": {
                    "enabled": allow_ring_slit,
                    "note": "고리형 성분을 절개선 단일 폴리곤으로 표현. 구멍은 배경으로 보존",
                },
                "affgrasp_repeat": affgrasp_repeat,
            },
            "input_manifests": [
                {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in manifest_paths
            ],
            "source_records": len(records),
            "counts": dict(sorted(counters.items())),
            "split_images": dict(sorted(split_images.items())),
            "class_instances": dict(sorted(class_instances.items())),
        }
        if not dry_run:
            dataset = {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "grasp_region", 1: "functional_region"},
            }
            (staging / "dataset.yaml").write_text(
                yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            _write_jsonl(staging / "export_manifest.jsonl", export_records)
            (staging / "dataset_version.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.replace(output_root)
        return export_records, summary
    except BaseException:
        if not dry_run and staging.exists():
            shutil.rmtree(staging)
        raise
