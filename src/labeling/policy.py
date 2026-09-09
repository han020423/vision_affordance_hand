"""변경 금지 프로젝트 라벨 정책과 마스크 변환 함수."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


BACKGROUND_VALUE: Final[int] = 0
GRASP_STORAGE_VALUE: Final[int] = 1
FUNCTIONAL_STORAGE_VALUE: Final[int] = 2
IGNORE_VALUE: Final[int] = 255

MODEL_CLASS_IDS: Final[dict[str, int]] = {
    "grasp_region": 0,
    "functional_region": 1,
}

AFFGRASP_SOURCE_NAMES: Final[dict[int, str]] = {
    0: "background",
    128: "graspable",
    255: "functional",
}

UMD_SOURCE_NAMES: Final[dict[int, str]] = {
    0: "background",
    1: "grasp",
    2: "cut",
    3: "scoop",
    4: "contain",
    5: "pound",
    6: "support",
    7: "wrap-grasp",
}

UMD_TO_STORAGE: Final[dict[int, int]] = {
    0: BACKGROUND_VALUE,
    1: GRASP_STORAGE_VALUE,
    2: FUNCTIONAL_STORAGE_VALUE,
    3: FUNCTIONAL_STORAGE_VALUE,
    4: IGNORE_VALUE,
    5: FUNCTIONAL_STORAGE_VALUE,
    6: FUNCTIONAL_STORAGE_VALUE,
    7: GRASP_STORAGE_VALUE,
}


@dataclass(frozen=True)
class ConversionSummary:
    """마스크 하나를 변환하면서 확인한 라벨 요약."""

    original_labels: tuple[str, ...]
    mapped_labels: tuple[str, ...]
    unknown_source_values: tuple[int, ...]
    ignore_reasons: tuple[str, ...]


def _observed_names(mask: np.ndarray, names: dict[int, str]) -> tuple[str, ...]:
    return tuple(names.get(int(value), f"unknown:{int(value)}") for value in np.unique(mask))


def convert_affgrasp_mask(mask: np.ndarray) -> tuple[np.ndarray, ConversionSummary]:
    """모호한 픽셀을 임의로 바꾸지 않고 Aff-Grasp 공식 마스크 값을 매핑한다."""

    source = np.asarray(mask)
    converted = np.zeros(source.shape, dtype=np.uint8)
    converted[source == 128] = GRASP_STORAGE_VALUE
    converted[source == 255] = FUNCTIONAL_STORAGE_VALUE

    unknown = tuple(
        int(value) for value in np.unique(source) if int(value) not in AFFGRASP_SOURCE_NAMES
    )
    if unknown:
        converted[np.isin(source, unknown)] = IGNORE_VALUE

    mapped = _mapped_names(converted)
    reasons = ("unknown_affgrasp_value",) if unknown else ()
    return converted, ConversionSummary(
        original_labels=_observed_names(source, AFFGRASP_SOURCE_NAMES),
        mapped_labels=mapped,
        unknown_source_values=unknown,
        ignore_reasons=reasons,
    )


def convert_umd_mask(mask: np.ndarray) -> tuple[np.ndarray, ConversionSummary]:
    """`contain`과 알 수 없는 값을 ignore로 유지하면서 UMD 라벨을 매핑한다."""

    source = np.asarray(mask)
    converted = np.full(source.shape, IGNORE_VALUE, dtype=np.uint8)
    for source_value, storage_value in UMD_TO_STORAGE.items():
        converted[source == source_value] = storage_value

    observed = {int(value) for value in np.unique(source)}
    unknown = tuple(sorted(observed.difference(UMD_SOURCE_NAMES)))
    reasons: list[str] = []
    if 4 in observed:
        reasons.append("umd_contain_policy")
    if unknown:
        reasons.append("unknown_umd_value")

    return converted, ConversionSummary(
        original_labels=_observed_names(source, UMD_SOURCE_NAMES),
        mapped_labels=_mapped_names(converted),
        unknown_source_values=unknown,
        ignore_reasons=tuple(reasons),
    )


def _mapped_names(mask: np.ndarray) -> tuple[str, ...]:
    names = {
        BACKGROUND_VALUE: "background",
        GRASP_STORAGE_VALUE: "grasp_region",
        FUNCTIONAL_STORAGE_VALUE: "functional_region",
        IGNORE_VALUE: "ignore",
    }
    return tuple(names[int(value)] for value in np.unique(mask))
