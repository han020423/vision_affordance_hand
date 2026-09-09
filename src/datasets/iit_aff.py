"""IIT-AFF(2017) 데이터셋 파서와 통합 라벨 변환.

IIT-AFF는 실제 어수선한 장면 8,835장에 픽셀 단위 affordance 라벨(txt 행렬)을
제공한다. 장면 단위 데이터라 실제 물체 인스턴스 ID가 없으므로, Aff-Grasp
전례를 따라 모든 샘플을 `pretrain`(train 전용)으로 두고 validation/test에는
사용하지 않는다.

라벨 매핑 (사용자 승인으로 추가된 데이터셋, 2026-08-19):
  grasp(5), w-grasp(9)                     → grasp_region(저장값 1)
  cut(2), display(3), engine(4), hit(6),
  pound(7), support(8)                     → functional_region(저장값 2)
  contain(1)                               → ignore(255)  ※ UMD contain 정책과 동일
  알 수 없는 값                            → ignore(255) + human_review
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.labeling.components import connected_components
from src.labeling.policy import (
    BACKGROUND_VALUE,
    ConversionSummary,
    FUNCTIONAL_STORAGE_VALUE,
    GRASP_STORAGE_VALUE,
    IGNORE_VALUE,
)

from .common import SampleRecord, relative_or_absolute, save_mask

IIT_SOURCE_NAMES: dict[int, str] = {
    0: "background",
    1: "contain",
    2: "cut",
    3: "display",
    4: "engine",
    5: "grasp",
    6: "hit",
    7: "pound",
    8: "support",
    9: "w-grasp",
}

# 파지 종류 재매핑에서 사용하는 원본 값: 5=grasp(손잡이형), 9=w-grasp(몸통형)
IIT_HANDLE_SOURCE_VALUE = 5
IIT_BODY_SOURCE_VALUE = 9

IIT_TO_STORAGE: dict[int, int] = {
    0: BACKGROUND_VALUE,
    1: IGNORE_VALUE,               # contain: 컵·용기 정책 충돌 방지
    2: FUNCTIONAL_STORAGE_VALUE,   # cut
    3: FUNCTIONAL_STORAGE_VALUE,   # display: 화면은 회피 대상 기능부로 취급
    4: FUNCTIONAL_STORAGE_VALUE,   # engine
    5: GRASP_STORAGE_VALUE,        # grasp
    6: FUNCTIONAL_STORAGE_VALUE,   # hit
    7: FUNCTIONAL_STORAGE_VALUE,   # pound
    8: FUNCTIONAL_STORAGE_VALUE,   # support
    9: GRASP_STORAGE_VALUE,        # w-grasp
}


def load_iit_label_matrix(path: Path) -> np.ndarray:
    """공백 구분 정수 행렬 txt 라벨을 빠르게 읽는다."""

    text = path.read_text(encoding="ascii")
    first_line = text.split("\n", 1)[0]
    width = len(first_line.split())
    values = np.fromstring(text, dtype=np.int16, sep=" ")
    if width == 0 or values.size % width:
        raise ValueError(f"라벨 행렬 형태가 올바르지 않습니다: {path}")
    return values.reshape(-1, width)


def convert_iit_mask(mask: np.ndarray) -> tuple[np.ndarray, ConversionSummary]:
    """contain과 알 수 없는 값을 ignore로 유지하면서 IIT-AFF 라벨을 매핑한다."""

    source = np.asarray(mask)
    converted = np.full(source.shape, IGNORE_VALUE, dtype=np.uint8)
    for source_value, storage_value in IIT_TO_STORAGE.items():
        converted[source == source_value] = storage_value

    observed = {int(value) for value in np.unique(source)}
    unknown = tuple(sorted(observed.difference(IIT_SOURCE_NAMES)))
    reasons: list[str] = []
    if 1 in observed:
        reasons.append("iit_contain_policy")
    if unknown:
        reasons.append("unknown_iit_value")

    storage_names = {
        BACKGROUND_VALUE: "background",
        GRASP_STORAGE_VALUE: "grasp_region",
        FUNCTIONAL_STORAGE_VALUE: "functional_region",
        IGNORE_VALUE: "ignore",
    }
    return converted, ConversionSummary(
        original_labels=tuple(
            IIT_SOURCE_NAMES.get(int(value), f"unknown:{int(value)}")
            for value in np.unique(source)
        ),
        mapped_labels=tuple(storage_names[int(value)] for value in np.unique(converted)),
        unknown_source_values=unknown,
        ignore_reasons=tuple(reasons),
    )


def discover_iit_aff(root: Path) -> list[tuple[str, Path, Path]]:
    """`rgb/*.jpg`와 `affordances_labels/*.txt` 쌍을 항상 같은 순서로 찾는다."""

    rgb_root = root / "rgb"
    label_root = root / "affordances_labels"
    if not rgb_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"IIT-AFF rgb/affordances_labels 디렉터리가 없습니다: {root}")
    samples: list[tuple[str, Path, Path]] = []
    for image_path in sorted(rgb_root.glob("*.jpg")):
        source_id = image_path.stem
        samples.append((source_id, image_path, label_root / f"{source_id}.txt"))
    return samples


def convert_iit_aff(
    root: Path,
    output_root: Path,
    repository_root: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> list[SampleRecord]:
    """IIT-AFF 라벨을 통합 형식으로 변환하고 분리 연결 성분을 보존한다."""

    records: list[SampleRecord] = []
    samples = discover_iit_aff(root)
    if limit is not None:
        samples = samples[:limit]

    for source_id, image_path, label_path in samples:
        semantic_path = output_root / "semantic" / f"{source_id}.png"
        instance_path = output_root / "instances" / f"{source_id}.png"
        record = SampleRecord(
            source_dataset="iit_aff",
            source_id=source_id,
            # 장면 데이터라 실제 물체 ID가 없다. 임의 분할 방지를 위해 명시한다.
            object_id=f"not_provided:{source_id}",
            object_id_status="not_provided_by_source",
            image_path=relative_or_absolute(image_path, repository_root),
            source_mask_path=relative_or_absolute(label_path, repository_root),
            semantic_mask_path=relative_or_absolute(semantic_path, repository_root),
            instance_mask_path=relative_or_absolute(instance_path, repository_root),
            split="pretrain",
        )

        if not label_path.is_file():
            record.conversion_status = "excluded"
            record.reasons.append("missing_mask")
            records.append(record)
            continue

        try:
            with Image.open(image_path) as image:
                image_size = image.size
            source_mask = load_iit_label_matrix(label_path)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            record.conversion_status = "excluded"
            record.reasons.append(f"unreadable_file:{type(exc).__name__}")
            records.append(record)
            continue

        if image_size != (source_mask.shape[1], source_mask.shape[0]):
            record.conversion_status = "excluded"
            record.reasons.append(
                f"size_mismatch:image={image_size},mask={source_mask.shape[::-1]}"
            )
            records.append(record)
            continue

        converted, summary = convert_iit_mask(source_mask)
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
