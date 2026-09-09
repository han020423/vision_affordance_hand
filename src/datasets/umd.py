"""UMD 공식 RGB-D Part Affordance 도구 데이터셋 파서."""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat

from src.labeling.components import connected_components
from src.labeling.policy import convert_umd_mask

from .common import SampleRecord, relative_or_absolute, save_mask


_RGB_SUFFIX = "_rgb.jpg"
ProgressCallback = Callable[[str, int, int], None]


def object_id_from_path(image_path: Path) -> str:
    """공식 물체 디렉터리 이름을 실제 물체 ID로 사용한다."""

    return image_path.parent.name


def frame_number(source_id: str) -> int | None:
    """다른 숫자를 추측하지 않고 파일명 끝의 프레임 번호만 읽는다."""

    match = re.search(r"(\d+)$", source_id)
    return int(match.group(1)) if match else None


def discover_umd(tools_root: Path) -> list[tuple[str, Path, Path]]:
    """RGB와 most-likely-affordance MAT 파일 쌍을 하위 폴더까지 찾는다."""

    if not tools_root.is_dir():
        raise FileNotFoundError(f"UMD tools 디렉터리가 없습니다: {tools_root}")
    samples: list[tuple[str, Path, Path]] = []
    for image_path in sorted(tools_root.rglob(f"*{_RGB_SUFFIX}")):
        stem = image_path.name.removesuffix(_RGB_SUFFIX)
        source_id = f"{image_path.parent.name}/{stem}"
        samples.append((source_id, image_path, image_path.with_name(f"{stem}_label.mat")))
    return samples


def ground_truth_type(mat: dict[str, np.ndarray]) -> str | None:
    """공식 MAT 파일에 `gt_type`이 있으면 해당 값을 반환한다."""

    if "gt_type" not in mat:
        return None
    values = np.asarray(mat["gt_type"]).reshape(-1)
    if values.size != 1:
        return None
    return str(values[0]).strip().lower()


def _read_gt_type(path_text: str) -> str:
    """프로세스 풀에서 안전하게 UMD 품질 메타데이터만 읽는다."""

    path = Path(path_text)
    if not path.is_file():
        return "missing_mask"
    try:
        mat = loadmat(path, variable_names=("gt_type",))
    except (OSError, ValueError):
        return "unreadable_mask"
    return ground_truth_type(mat) or "missing_or_invalid_gt_type"


def index_umd_quality(
    samples: list[tuple[str, Path, Path]],
    *,
    jobs: int,
    progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """조밀 라벨을 불러오지 않고 공식 수동/자동 메타데이터를 색인한다."""

    if jobs < 1:
        raise ValueError("jobs는 1 이상이어야 합니다")
    paths = [str(mask_path) for _, _, mask_path in samples]
    if jobs == 1:
        quality_values = map(_read_gt_type, paths)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        quality_values = executor.map(_read_gt_type, paths, chunksize=64)

    result: dict[str, str] = {}
    try:
        total = len(samples)
        for index, ((source_id, _, _), quality) in enumerate(
            zip(samples, quality_values), start=1
        ):
            result[source_id] = quality
            if progress is not None and (index % 1000 == 0 or index == total):
                progress("quality_index", index, total)
    finally:
        if jobs != 1:
            executor.shutdown()
    return result


def _allocate_object_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """가능하면 모든 물체를 유지하면서 정확한 목표 수를 비례 배분한다."""

    total = sum(counts.values())
    if target <= 0:
        raise ValueError("target_manual_frames는 양수여야 합니다")
    if target >= total:
        return dict(counts)

    quotas = {object_id: 0 for object_id in counts}
    remaining = target
    capacities = dict(counts)
    if target >= len(counts):
        for object_id in counts:
            quotas[object_id] = 1
            capacities[object_id] -= 1
        remaining -= len(counts)

    capacity_total = sum(capacities.values())
    if remaining == 0 or capacity_total == 0:
        return quotas

    exact = {
        object_id: remaining * capacity / capacity_total
        for object_id, capacity in capacities.items()
    }
    for object_id in sorted(counts):
        addition = min(capacities[object_id], int(exact[object_id]))
        quotas[object_id] += addition
    left = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda object_id: (-(exact[object_id] - int(exact[object_id])), object_id),
    )
    for object_id in order:
        if left == 0:
            break
        if quotas[object_id] < counts[object_id]:
            quotas[object_id] += 1
            left -= 1
    if left != 0:
        raise RuntimeError(f"UMD 프레임 {target}장을 배분하지 못했습니다. 남은 수: {left}")
    return quotas


def deterministic_manual_subsample(
    samples: list[tuple[str, Path, Path]],
    quality_by_source_id: dict[str, str],
    target_manual_frames: int | None,
    *,
    manual_frame_modulo: int | None,
) -> set[str]:
    """물체별 비례 할당량에 맞춰 시간 간격이 떨어진 수동 프레임을 선택한다."""

    by_object: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for sample in samples:
        source_id, image_path, _ = sample
        quality = quality_by_source_id[source_id]
        inferred_manual = (
            quality == "missing_or_invalid_gt_type"
            and manual_frame_modulo is not None
            and frame_number(source_id) is not None
            and frame_number(source_id) % manual_frame_modulo == 0
        )
        if quality == "manual" or inferred_manual:
            by_object[object_id_from_path(image_path)].append(sample)

    for object_samples in by_object.values():
        object_samples.sort(
            key=lambda sample: (
                frame_number(sample[0]) if frame_number(sample[0]) is not None else -1,
                sample[0],
            )
        )
    counts = {object_id: len(object_samples) for object_id, object_samples in by_object.items()}
    if not counts:
        return set()
    target = sum(counts.values()) if target_manual_frames is None else target_manual_frames
    quotas = _allocate_object_quotas(counts, target)

    selected: set[str] = set()
    for object_id in sorted(by_object):
        object_samples = by_object[object_id]
        quota = quotas[object_id]
        if quota == 0:
            continue
        if quota == 1:
            indices = [len(object_samples) // 2]
        elif quota == len(object_samples):
            indices = list(range(len(object_samples)))
        else:
            indices = [
                round(index * (len(object_samples) - 1) / (quota - 1))
                for index in range(quota)
            ]
        if len(set(indices)) != quota:
            raise RuntimeError(f"시간 간격 선택 결과가 중복되었습니다: {object_id}")
        selected.update(object_samples[index][0] for index in indices)
    return selected


def convert_umd(
    tools_root: Path,
    output_root: Path,
    repository_root: Path,
    split_by_object: dict[str, str],
    *,
    manual_frame_modulo: int | None,
    target_manual_frames: int | None = None,
    jobs: int = 1,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> list[SampleRecord]:
    """물체 ID 분할과 `contain -> ignore` 정책을 적용해 UMD 마스크를 변환한다."""

    records: list[SampleRecord] = []
    samples = discover_umd(tools_root)
    if limit is not None:
        samples = samples[:limit]

    quality_by_source_id = index_umd_quality(samples, jobs=jobs, progress=progress)
    selected_source_ids = deterministic_manual_subsample(
        samples,
        quality_by_source_id,
        target_manual_frames,
        manual_frame_modulo=manual_frame_modulo,
    )

    for sample_index, (source_id, image_path, mask_path) in enumerate(samples, start=1):
        if progress is not None and (
            sample_index % 1000 == 0 or sample_index == len(samples)
        ):
            progress("conversion", sample_index, len(samples))
        safe_id = source_id.replace("/", "__")
        semantic_path = output_root / "semantic" / f"{safe_id}.png"
        instance_path = output_root / "instances" / f"{safe_id}.png"
        object_id = object_id_from_path(image_path)
        split = split_by_object.get(object_id, "unassigned")
        record = SampleRecord(
            source_dataset="umd",
            source_id=source_id,
            object_id=object_id,
            object_id_status="official_directory_name",
            image_path=relative_or_absolute(image_path, repository_root),
            source_mask_path=relative_or_absolute(mask_path, repository_root),
            semantic_mask_path=relative_or_absolute(semantic_path, repository_root),
            instance_mask_path=relative_or_absolute(instance_path, repository_root),
            split=split,
        )

        if split == "unassigned":
            record.reasons.append("object_split_unassigned")
        if not mask_path.is_file():
            record.conversion_status = "excluded"
            record.reasons.append("missing_mask")
            records.append(record)
            continue

        gt_type = quality_by_source_id[source_id]
        if gt_type == "automatic":
            record.conversion_status = "excluded"
            record.selection_status = "ineligible"
            record.reasons.append("automatic_label_excluded")
            records.append(record)
            continue
        if gt_type != "manual":
            if manual_frame_modulo is None or not (
                frame_number(source_id) is not None
                and frame_number(source_id) % manual_frame_modulo == 0
            ):
                record.conversion_status = "excluded"
                record.selection_status = "ineligible"
                record.reasons.append("label_quality_unknown")
                records.append(record)
                continue
            record.reasons.append("manual_gt_inferred_from_configured_modulo")

        if source_id not in selected_source_ids:
            record.conversion_status = "excluded"
            record.selection_status = "not_selected"
            record.reasons.append("deterministic_temporal_subsample_excluded")
            records.append(record)
            continue
        record.selection_status = "selected"

        try:
            mat = loadmat(mask_path, variable_names=("gt_label",))
            if "gt_label" not in mat:
                raise KeyError("gt_label")
            source_mask = np.asarray(mat["gt_label"]).squeeze()
        except (OSError, UnidentifiedImageError, ValueError, KeyError) as exc:
            record.conversion_status = "excluded"
            record.reasons.append(f"unreadable_or_invalid_mask:{type(exc).__name__}")
            records.append(record)
            continue

        try:
            with Image.open(image_path) as image:
                image_size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            record.conversion_status = "excluded"
            record.reasons.append(f"unreadable_image:{type(exc).__name__}")
            records.append(record)
            continue

        if source_mask.ndim != 2:
            record.conversion_status = "excluded"
            record.reasons.append(f"invalid_mask_shape:{source_mask.shape}")
            records.append(record)
            continue
        mask_size = (int(source_mask.shape[1]), int(source_mask.shape[0]))
        if image_size != mask_size:
            record.conversion_status = "excluded"
            record.reasons.append(f"size_mismatch:image={image_size},mask={mask_size}")
            records.append(record)
            continue

        converted, summary = convert_umd_mask(source_mask)
        instance_mask, components = connected_components(converted)
        record.original_labels = list(summary.original_labels)
        record.mapped_labels = list(summary.mapped_labels)
        record.reasons.extend(summary.ignore_reasons)
        record.components = [component.to_dict() for component in components]

        if not components:
            record.conversion_status = "excluded"
            record.reasons.append("empty_affordance_mask")
        elif summary.unknown_source_values or split == "unassigned":
            record.conversion_status = "human_review"
        else:
            record.conversion_status = "converted"

        if not dry_run and record.conversion_status != "excluded":
            save_mask(semantic_path, converted)
            save_mask(instance_path, instance_mask)
        records.append(record)

    return records
