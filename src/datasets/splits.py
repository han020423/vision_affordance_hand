"""물체 단위 데이터 분할을 읽고 누수를 검증한다."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import yaml


VALID_SPLITS = {"train", "validation", "test", "pretrain", "external_evaluation"}
_UMD_FOLD_PATTERN = re.compile(
    r"^folds:\s+((?:[01]\s+){9}[01])\s+object:\s+(\S+)\s*$"
)


def load_object_splits(path: Path | None) -> dict[str, str]:
    """명시된 물체 ID를 읽으며 파일이 없으면 분할 배정을 반환하지 않는다."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    result: dict[str, str] = {}
    for split, object_ids in document.get("splits", {}).items():
        if split not in VALID_SPLITS:
            raise ValueError(f"지원하지 않는 분할 {split!r}입니다: {path}")
        for object_id in object_ids or []:
            object_id = str(object_id)
            if object_id in result:
                raise ValueError(
                    f"물체 {object_id!r}가 {result[object_id]!r}와 {split!r} 분할에 모두 포함되어 있습니다"
                )
            result[object_id] = split
    return result


def parse_umd_official_folds(path: Path) -> dict[str, frozenset[int]]:
    """UMD 공식 10-fold 물체 소속 파일을 해석한다."""

    memberships: dict[str, frozenset[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _UMD_FOLD_PATTERN.match(line)
            if match is None:
                raise ValueError(f"잘못된 UMD fold 행입니다({path}:{line_number}): {line!r}")
            flags = [int(value) for value in match.group(1).split()]
            object_id = match.group(2)
            if object_id in memberships:
                raise ValueError(f"UMD 물체 {object_id!r}가 중복되었습니다: {path}")
            memberships[object_id] = frozenset(
                index for index, enabled in enumerate(flags) if enabled
            )
    if not memberships:
        raise ValueError(f"UMD fold 행을 찾지 못했습니다: {path}")
    return memberships


def build_umd_object_splits(
    fold_path: Path,
    object_ids: set[str],
    *,
    validation_fold: int,
    test_fold: int,
) -> dict[str, str]:
    """선택한 UMD 공식 fold 두 개로 서로 겹치지 않는 분할을 만든다."""

    if validation_fold == test_fold:
        raise ValueError("validation fold와 test fold는 서로 달라야 합니다")
    if not 0 <= validation_fold <= 9 or not 0 <= test_fold <= 9:
        raise ValueError("UMD fold 번호는 0부터 9 사이여야 합니다")

    memberships = parse_umd_official_folds(fold_path)
    official_ids = set(memberships)
    if official_ids != object_ids:
        missing = sorted(object_ids.difference(official_ids))
        extra = sorted(official_ids.difference(object_ids))
        raise ValueError(f"UMD fold와 물체 디렉터리가 일치하지 않습니다: 누락={missing}, 추가={extra}")

    result: dict[str, str] = {}
    for object_id in sorted(object_ids):
        folds = memberships[object_id]
        in_validation = validation_fold in folds
        in_test = test_fold in folds
        if in_validation and in_test:
            raise ValueError(
                f"물체 {object_id!r}가 선택한 validation과 test fold에 모두 포함되어 있습니다"
            )
        if in_test:
            result[object_id] = "test"
        elif in_validation:
            result[object_id] = "validation"
        else:
            result[object_id] = "train"
    return result


def find_split_leakage(records: list[dict[str, object]]) -> dict[str, list[str]]:
    """실제 적용 분할이 둘 이상인 물체 ID를 반환한다."""

    by_object: dict[str, set[str]] = defaultdict(set)
    for record in records:
        object_id = str(record.get("object_id", ""))
        split = str(record.get("split", "unassigned"))
        if object_id and split != "unassigned":
            by_object[object_id].add(split)
    return {
        object_id: sorted(splits)
        for object_id, splits in by_object.items()
        if len(splits) > 1
    }
