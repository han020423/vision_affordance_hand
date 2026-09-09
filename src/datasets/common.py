"""데이터 변환에서 공통으로 사용하는 기록 구조와 파일 도우미."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class SampleRecord:
    """발견한 모든 샘플에 반드시 남기는 검증 가능한 manifest 항목."""

    source_dataset: str
    source_id: str
    object_id: str
    object_id_status: str
    image_path: str
    source_mask_path: str
    semantic_mask_path: str | None
    instance_mask_path: str | None
    original_labels: list[str] = field(default_factory=list)
    mapped_labels: list[str] = field(default_factory=list)
    split: str = "unassigned"
    conversion_status: str = "pending"
    selection_status: str | None = None
    reasons: list[str] = field(default_factory=list)
    components: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def relative_or_absolute(path: Path, base: Path) -> str:
    """파일이 저장소 안에 있으면 다른 환경에서도 쓸 수 있는 상대경로를 사용한다."""

    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def save_mask(path: Path, mask: np.ndarray) -> None:
    """필요한 상위 디렉터리를 만든 뒤 무손실 PNG 마스크를 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype == np.uint16:
        Image.fromarray(mask, mode="I;16").save(path)
    else:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def write_jsonl(path: Path, records: Iterable[SampleRecord]) -> None:
    """항상 같은 순서의 JSON Lines 결과를 원자적으로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
