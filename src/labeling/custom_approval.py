"""사람이 전체 검수한 자체 머그 마스크를 승인 데이터로 승격한다."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.datasets.common import relative_or_absolute, save_mask
from src.labeling.custom_review import (
    BODY_INSTANCE_ID,
    FUNCTIONAL_INSTANCE_ID,
    HANDLE_INSTANCE_ID,
    inspect_mug_instance_mask,
    remove_tiny_instance_fragments,
    render_review_overlay,
    semantic_from_instances,
)

# 카테고리별 기능 부위 이름. 2026-08-21 승인 정책과 같다.
FUNCTIONAL_PART_BY_CATEGORY = {"scissors": "blade", "screwdriver": "shaft"}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """JSONL을 읽되 손상된 행 번호를 포함해 오류를 보고한다."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON 오류: {error}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """manifest를 UTF-8과 정렬된 키로 저장한다."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fixed_instance_components(instance_mask: np.ndarray) -> list[dict[str, object]]:
    """손잡이=1, 몸통=2 인스턴스의 YOLO 변환용 메타데이터를 만든다."""

    components: list[dict[str, object]] = []
    height, width = instance_mask.shape
    for instance_id, part in (
        (HANDLE_INSTANCE_ID, "handle"),
        (BODY_INSTANCE_ID, "body"),
    ):
        ys, xs = np.where(instance_mask == instance_id)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        components.append(
            {
                "component_id": instance_id,
                "class_name": "grasp_region",
                "model_class_id": 0,
                "storage_value": 1,
                "pixel_count": int(len(xs)),
                "bbox_xywh": [
                    x_min,
                    y_min,
                    x_max - x_min + 1,
                    y_max - y_min + 1,
                ],
                "touches_boundary": (
                    x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1
                ),
                "part": part,
            }
        )
    return components


def category_instance_components(
    instance_mask: np.ndarray, category: str
) -> list[dict[str, object]]:
    """카테고리 규칙에 맞는 인스턴스별 변환용 메타데이터를 만든다.

    손잡이(1)·몸통(2)은 grasp_region, 기능(3)은 functional_region이다.
    가위 고리처럼 한 인스턴스 값이 여러 연결 성분이면 이후 변환 단계가
    성분 단위로 다시 나누므로 여기서는 값 단위로 기록한다.
    """

    part_by_id = {
        HANDLE_INSTANCE_ID: ("handle", "grasp_region", 0, 1),
        BODY_INSTANCE_ID: ("body", "grasp_region", 0, 1),
        FUNCTIONAL_INSTANCE_ID: (
            FUNCTIONAL_PART_BY_CATEGORY.get(category, "functional"),
            "functional_region",
            1,
            2,
        ),
    }
    components: list[dict[str, object]] = []
    height, width = instance_mask.shape
    for instance_id, (part, class_name, model_class_id, storage_value) in part_by_id.items():
        ys, xs = np.where(instance_mask == instance_id)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        components.append(
            {
                "component_id": instance_id,
                "class_name": class_name,
                "model_class_id": model_class_id,
                "storage_value": storage_value,
                "pixel_count": int(len(xs)),
                "bbox_xywh": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                "touches_boundary": (
                    x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1
                ),
                "part": part,
            }
        )
    return components


def inspect_category_instance_mask(
    instance_mask: np.ndarray,
    category: str,
    *,
    minimum_pixels: int = 64,
) -> list[str]:
    """카테고리 규칙 위반과 검수 참고 사항을 표시 목록으로 반환한다."""

    allowed_by_category = {
        "mug": {0, HANDLE_INSTANCE_ID, BODY_INSTANCE_ID},
        "insulated_mug": {0, HANDLE_INSTANCE_ID, BODY_INSTANCE_ID},
        "scissors": {0, HANDLE_INSTANCE_ID, FUNCTIONAL_INSTANCE_ID},
        "screwdriver": {0, HANDLE_INSTANCE_ID, FUNCTIONAL_INSTANCE_ID},
    }
    allowed = allowed_by_category.get(category)
    if allowed is None:
        raise ValueError(f"승격 규칙이 정의되지 않은 카테고리입니다: {category}")
    values = set(np.unique(instance_mask).tolist())
    if not values <= allowed:
        raise ValueError(
            f"{category}에 허용되지 않은 인스턴스 값입니다: {sorted(values - allowed)}"
        )

    flags: list[str] = []
    if category in ("mug", "insulated_mug"):
        _, flags = inspect_mug_instance_mask(instance_mask, minimum_pixels=minimum_pixels)
        return flags

    if not np.any(instance_mask == FUNCTIONAL_INSTANCE_ID):
        raise ValueError(f"{category} 승인 마스크에 기능 부위가 없습니다")
    handle_binary = (instance_mask == HANDLE_INSTANCE_ID).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(handle_binary, connectivity=8)
    handle_components = sum(
        1
        for local_id in range(1, count)
        if int(stats[local_id, cv2.CC_STAT_AREA]) >= minimum_pixels
    )
    if handle_components == 0:
        flags.append("손잡이_비가시_또는_라벨없음")
    if category == "scissors" and handle_components not in (0, 2):
        flags.append("가위_고리_성분수_확인")

    foreground = instance_mask > 0
    if np.any(foreground):
        if (
            np.any(foreground[0])
            or np.any(foreground[-1])
            or np.any(foreground[:, 0])
            or np.any(foreground[:, -1])
        ):
            flags.append("마스크_영상경계_접촉")
        if float(np.count_nonzero(foreground)) / foreground.size > 0.7:
            flags.append("마스크_면적_과다")
    return sorted(set(flags))


def _read_image_required(path: Path, mode: int) -> np.ndarray:
    """OpenCV 입력 실패 시 경로를 포함해 중단한다."""

    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def _write_image_required(path: Path, image: np.ndarray) -> None:
    """OpenCV 출력 실패를 조용히 무시하지 않는다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")


def _assert_no_split_leakage(rows: Iterable[dict[str, object]]) -> None:
    """같은 실제 물체가 둘 이상의 분할에 들어가면 승격을 중단한다."""

    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits[str(row["object_id"])].add(str(row["split"]))
    leakage = {object_id: values for object_id, values in splits.items() if len(values) > 1}
    if leakage:
        raise ValueError(f"자체 승인 데이터에 물체 ID 분할 누수가 있습니다: {leakage}")


def promote_new132_review(
    workspace: Path,
    source_manifest_path: Path,
    approved_v1_manifest_path: Path,
    output_root: Path,
    repository_root: Path,
    *,
    minimum_fragment_pixels: int = 64,
) -> dict[str, object]:
    """2026-08-21 추가 촬영 132장 검수본을 승인 v2로 승격한다.

    - 기존 승인 v1의 160 레코드는 파일을 복사하지 않고 그대로 이어받는다
      (v1 폴더는 불변으로 보존하며 v2 manifest가 v1 마스크 경로를 참조한다).
    - 신규 132장은 카테고리 규칙(머그=손잡이/몸통, 가위·드라이버=손잡이/기능)을
      검증한 뒤 v2 마스크·오버레이로 저장한다.
    """

    if output_root.exists():
        raise FileExistsError(f"기존 승인 데이터셋을 덮어쓰지 않습니다: {output_root}")
    review_rows = read_jsonl(workspace / "review_manifest.jsonl")
    source_rows = read_jsonl(source_manifest_path)
    v1_rows = read_jsonl(approved_v1_manifest_path)

    bad_status = sorted(
        {
            str(row.get("review_status"))
            for row in review_rows
            if row.get("review_status") not in ("human_corrected", "excluded_ambiguous")
        }
    )
    if bad_status:
        raise ValueError(f"검수가 끝나지 않은 상태가 있습니다: {bad_status}")

    review_by_source = {str(row["source_id"]): row for row in review_rows}
    v1_source_ids = {str(row["source_id"]) for row in v1_rows}
    new_sources = [
        row
        for row in source_rows
        if str(row["source_id"]) not in v1_source_ids
    ]
    missing_review = sorted(
        str(row["source_id"])
        for row in new_sources
        if str(row["source_id"]) not in review_by_source
    )
    if missing_review:
        raise ValueError(f"검수 기록이 없는 신규 이미지가 있습니다: {missing_review[:5]}")

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 승인 폴더를 먼저 확인하세요: {staging}")
    for directory in ("semantic_masks", "instance_masks", "overlays"):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    approved_rows: list[dict[str, object]] = [dict(row) for row in v1_rows]
    removed_pixels = Counter()
    validation_flags = Counter()
    excluded_rows: list[str] = []
    part_name_by_id = {HANDLE_INSTANCE_ID: "handle", BODY_INSTANCE_ID: "body", FUNCTIONAL_INSTANCE_ID: "functional"}

    for source in new_sources:
        source_id = str(source["source_id"])
        object_id = str(source["object_id"])
        category = str(source.get("category", ""))
        review = review_by_source[source_id]
        if review.get("review_status") == "excluded_ambiguous":
            excluded_row = dict(source)
            excluded_row.update(
                {
                    "annotation_status": "excluded_ambiguous",
                    "conversion_status": "excluded",
                    "reasons": ["human_review_excluded_ambiguous"],
                    "label_version": "custom_approved_v2",
                }
            )
            approved_rows.append(excluded_row)
            excluded_rows.append(source_id)
            continue

        image_path = Path(str(source["image_path"]))
        if not image_path.is_absolute():
            image_path = repository_root / image_path
        image = _read_image_required(image_path, cv2.IMREAD_COLOR)
        mask_path = workspace / str(review["corrected_instance_path"])
        raw_instance = _read_image_required(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_instance.shape != image.shape[:2]:
            raise ValueError(f"승인 이미지와 마스크 크기가 다릅니다: {source_id}")
        instance, removed = remove_tiny_instance_fragments(
            raw_instance, minimum_pixels=minimum_fragment_pixels
        )
        for instance_id, count in removed.items():
            removed_pixels[part_name_by_id.get(instance_id, str(instance_id))] += count
        if category in ("mug", "insulated_mug") and not np.any(instance == BODY_INSTANCE_ID):
            raise ValueError(f"승인 마스크에 몸통이 없습니다: {source_id}")
        flags = inspect_category_instance_mask(
            instance, category, minimum_pixels=minimum_fragment_pixels
        )
        for flag in flags:
            validation_flags[flag] += 1
        semantic = semantic_from_instances(instance)
        components = category_instance_components(instance, category)

        stem = Path(source_id.split("/", maxsplit=1)[-1]).name
        semantic_target = staging / "semantic_masks" / object_id / f"{stem}.png"
        instance_target = staging / "instance_masks" / object_id / f"{stem}.png"
        overlay_target = staging / "overlays" / object_id / f"{stem}.jpg"
        save_mask(semantic_target, semantic)
        save_mask(instance_target, instance)
        _write_image_required(overlay_target, render_review_overlay(image, instance))

        approved = dict(source)
        approved.update(
            {
                "annotation_status": "human_verified",
                "conversion_status": "converted",
                "mapped_labels": sorted({str(c["class_name"]) for c in components}),
                "original_labels": ["custom_human_review"],
                "reasons": ["new132_human_verified"],
                "review_status": "human_approved_new132",
                "label_source": "human_corrected_with_foundation_model_initialization",
                "handle_visibility": review.get("handle_visibility"),
                "semantic_mask_path": relative_or_absolute(
                    output_root / semantic_target.relative_to(staging), repository_root
                ),
                "instance_mask_path": relative_or_absolute(
                    output_root / instance_target.relative_to(staging), repository_root
                ),
                "approved_overlay_path": relative_or_absolute(
                    output_root / overlay_target.relative_to(staging), repository_root
                ),
                "components": components,
                "validation_flags": flags,
                "functional_region_generated": bool(
                    np.any(instance == FUNCTIONAL_INSTANCE_ID)
                ),
                "label_version": "custom_approved_v2",
            }
        )
        approved_rows.append(approved)

    _assert_no_split_leakage(
        row for row in approved_rows if row.get("conversion_status") == "converted"
    )
    summary: dict[str, object] = {
        "status": "human_verified",
        "records": len(approved_rows),
        "carried_from_v1": len(v1_rows),
        "new_records": len(approved_rows) - len(v1_rows),
        "excluded_new_records": excluded_rows,
        "by_object": dict(Counter(str(row["object_id"]) for row in approved_rows)),
        "by_split": dict(Counter(str(row["split"]) for row in approved_rows)),
        "validation_flags": dict(sorted(validation_flags.items())),
        "tiny_fragment_pixels_removed": dict(removed_pixels),
        "classes": {"0": "grasp_region", "1": "functional_region"},
        "functional_categories": sorted(FUNCTIONAL_PART_BY_CATEGORY),
        "v1_masks_referenced_not_copied": True,
    }
    write_jsonl(staging / "manifest.jsonl", approved_rows)
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    return summary


def promote_full_review(
    workspace: Path,
    source_manifest_path: Path,
    output_root: Path,
    repository_root: Path,
    *,
    minimum_fragment_pixels: int = 64,
) -> dict[str, object]:
    """전체 검수 마스크 140장과 검증된 배경 20장을 승인 세트로 승격한다."""

    if output_root.exists():
        raise FileExistsError(f"기존 승인 데이터셋을 덮어쓰지 않습니다: {output_root}")
    review_rows = read_jsonl(workspace / "review_manifest.jsonl")
    source_rows = read_jsonl(source_manifest_path)
    if any(row.get("review_status") != "human_corrected" for row in review_rows):
        raise ValueError("전체 140장을 저장 완료한 뒤 승인 데이터로 승격할 수 있습니다")

    review_by_source = {str(row["source_id"]): row for row in review_rows}
    object_sources = {str(row["source_id"]) for row in source_rows if not row.get("is_negative")}
    if set(review_by_source) != object_sources:
        missing = sorted(object_sources - set(review_by_source))
        extra = sorted(set(review_by_source) - object_sources)
        raise ValueError(f"전체 검수와 원본 manifest ID가 다릅니다: missing={missing}, extra={extra}")

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 승인 폴더를 먼저 확인하세요: {staging}")
    for directory in ("semantic_masks", "instance_masks", "overlays"):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    approved_rows: list[dict[str, object]] = []
    removed_pixels = Counter()
    validation_flags = Counter()
    try:
        for source in source_rows:
            source_id = str(source["source_id"])
            object_id = str(source["object_id"])
            image_path = Path(str(source["image_path"]))
            if not image_path.is_absolute():
                image_path = repository_root / image_path
            image = _read_image_required(image_path, cv2.IMREAD_COLOR)
            stem = Path(source_id.split("/", maxsplit=1)[-1]).name
            semantic_target = staging / "semantic_masks" / object_id / f"{stem}.png"
            instance_target = staging / "instance_masks" / object_id / f"{stem}.png"
            overlay_target = staging / "overlays" / object_id / f"{stem}.jpg"

            approved = dict(source)
            if bool(source.get("is_negative", False)):
                semantic = np.zeros(image.shape[:2], dtype=np.uint8)
                instance = np.zeros(image.shape[:2], dtype=np.uint16)
                components: list[dict[str, object]] = []
                flags: list[str] = []
                approved.update(
                    {
                        "annotation_status": "verified_empty",
                        "conversion_status": "converted",
                        "mapped_labels": ["background"],
                        "original_labels": [],
                        "reasons": ["verified_background_negative"],
                        "review_status": "previously_verified_background",
                        "label_source": "human_verified_negative",
                    }
                )
            else:
                review = review_by_source[source_id]
                mask_path = workspace / str(review["corrected_instance_path"])
                raw_instance = _read_image_required(mask_path, cv2.IMREAD_UNCHANGED)
                if raw_instance.shape != image.shape[:2]:
                    raise ValueError(f"승인 이미지와 마스크 크기가 다릅니다: {source_id}")
                instance, removed = remove_tiny_instance_fragments(
                    raw_instance, minimum_pixels=minimum_fragment_pixels
                )
                for instance_id, count in removed.items():
                    removed_pixels[
                        "handle" if instance_id == HANDLE_INSTANCE_ID else "body"
                    ] += count
                if not np.any(instance == BODY_INSTANCE_ID):
                    raise ValueError(f"승인 마스크에 몸통이 없습니다: {source_id}")
                semantic = semantic_from_instances(instance)
                _, flags = inspect_mug_instance_mask(
                    instance, minimum_pixels=minimum_fragment_pixels
                )
                components = fixed_instance_components(instance)
                approved.update(
                    {
                        "annotation_status": "human_verified",
                        "conversion_status": "converted",
                        "mapped_labels": ["grasp_region"],
                        "original_labels": ["custom_human_review"],
                        "reasons": ["full_140_round2_human_verified"],
                        "review_status": "human_approved_full_round2",
                        "label_source": "human_corrected_with_foundation_model_initialization",
                        "handle_visibility": review.get("handle_visibility"),
                    }
                )
            for flag in flags:
                validation_flags[flag] += 1

            save_mask(semantic_target, semantic)
            save_mask(instance_target, instance)
            _write_image_required(overlay_target, render_review_overlay(image, instance))
            approved.update(
                {
                    "semantic_mask_path": relative_or_absolute(
                        output_root / semantic_target.relative_to(staging), repository_root
                    ),
                    "instance_mask_path": relative_or_absolute(
                        output_root / instance_target.relative_to(staging), repository_root
                    ),
                    "approved_overlay_path": relative_or_absolute(
                        output_root / overlay_target.relative_to(staging), repository_root
                    ),
                    "components": components,
                    "validation_flags": flags,
                    "functional_region_generated": False,
                    "label_version": "custom_approved_v1",
                }
            )
            approved_rows.append(approved)

        _assert_no_split_leakage(approved_rows)
        summary: dict[str, object] = {
            "status": "human_verified",
            "records": len(approved_rows),
            "object_images": sum(not bool(row.get("is_negative")) for row in approved_rows),
            "background_images": sum(bool(row.get("is_negative")) for row in approved_rows),
            "by_object": dict(Counter(str(row["object_id"]) for row in approved_rows)),
            "by_split": dict(Counter(str(row["split"]) for row in approved_rows)),
            "validation_flags": dict(sorted(validation_flags.items())),
            "tiny_fragment_pixels_removed": dict(removed_pixels),
            "classes": {"0": "grasp_region", "1": "functional_region"},
            "functional_region_generated": False,
            "reserved_test_object_id": "mug_04",
        }
        write_jsonl(staging / "manifest.jsonl", approved_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except BaseException:
        # 실패 시 임시 결과를 남겨 어느 샘플에서 중단됐는지 확인할 수 있게 한다.
        raise
    return summary
