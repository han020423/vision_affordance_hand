"""Aff-Grasp 공식 ego-training 배포본 파서."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.labeling.components import connected_components
from src.labeling.policy import convert_affgrasp_mask

from .common import SampleRecord, relative_or_absolute, save_mask


def discover_affgrasp(training_root: Path) -> list[tuple[str, Path, Path]]:
    """공식 `*-img.jpg`/`*-label.png` 쌍을 항상 같은 순서로 찾는다."""

    samples: list[tuple[str, Path, Path]] = []
    if not training_root.is_dir():
        raise FileNotFoundError(f"Aff-Grasp 학습 디렉터리가 없습니다: {training_root}")
    for image_path in sorted(training_root.glob("*-img.jpg")):
        source_id = image_path.name.removesuffix("-img.jpg")
        samples.append((source_id, image_path, training_root / f"{source_id}-label.png"))
    return samples


def convert_affgrasp(
    training_root: Path,
    output_root: Path,
    repository_root: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> list[SampleRecord]:
    """Aff-Grasp 학습 마스크를 변환하고 분리된 연결 성분을 각각 보존한다."""

    records: list[SampleRecord] = []
    samples = discover_affgrasp(training_root)
    if limit is not None:
        samples = samples[:limit]

    for source_id, image_path, mask_path in samples:
        semantic_path = output_root / "semantic" / f"{source_id}.png"
        instance_path = output_root / "instances" / f"{source_id}.png"
        record = SampleRecord(
            source_dataset="affgrasp",
            source_id=source_id,
            object_id=f"not_provided:{source_id}",
            object_id_status="not_provided_by_source",
            image_path=relative_or_absolute(image_path, repository_root),
            source_mask_path=relative_or_absolute(mask_path, repository_root),
            semantic_mask_path=relative_or_absolute(semantic_path, repository_root),
            instance_mask_path=relative_or_absolute(instance_path, repository_root),
            split="pretrain",
        )

        if not mask_path.is_file():
            record.conversion_status = "excluded"
            record.reasons.append("missing_mask")
            records.append(record)
            continue

        try:
            with Image.open(image_path) as image:
                image_size = image.size
            with Image.open(mask_path) as source_mask_image:
                source_mask = np.asarray(source_mask_image.convert("L"))
                mask_size = source_mask_image.size
        except (OSError, UnidentifiedImageError) as exc:
            record.conversion_status = "excluded"
            record.reasons.append(f"unreadable_file:{type(exc).__name__}")
            records.append(record)
            continue

        if image_size != mask_size:
            record.conversion_status = "excluded"
            record.reasons.append(f"size_mismatch:image={image_size},mask={mask_size}")
            records.append(record)
            continue

        converted, summary = convert_affgrasp_mask(source_mask)
        instance_mask, components = connected_components(converted)
        record.original_labels = list(summary.original_labels)
        record.mapped_labels = list(summary.mapped_labels)
        record.reasons.extend(summary.ignore_reasons)
        record.components = [component.to_dict() for component in components]

        if not components:
            record.conversion_status = "excluded"
            record.reasons.append("empty_affordance_mask")
        elif summary.unknown_source_values:
            record.conversion_status = "human_review"
        else:
            record.conversion_status = "converted"

        if not dry_run and record.conversion_status != "excluded":
            save_mask(semantic_path, converted)
            save_mask(instance_path, instance_mask)
        records.append(record)

    return records
