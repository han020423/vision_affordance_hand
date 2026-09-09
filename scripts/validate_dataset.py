#!/usr/bin/env python
"""변환 manifest와 물체 단위 데이터 분할의 격리 여부를 검증한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.splits import find_split_leakage  # noqa: E402


REQUIRED_FIELDS = {
    "source_dataset",
    "source_id",
    "object_id",
    "original_labels",
    "mapped_labels",
    "split",
    "conversion_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for path in args.manifests:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                missing = REQUIRED_FIELDS.difference(record)
                if missing:
                    errors.append(f"{path}:{line_number}: 필수 항목 누락 {sorted(missing)}")
                records.append(record)

    leakage = find_split_leakage(records)
    for object_id, splits in sorted(leakage.items()):
        errors.append(f"분할 누수: {object_id} -> {splits}")

    statuses = Counter(str(record.get("conversion_status")) for record in records)
    print(f"레코드 수: {len(records)}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    if errors:
        for error in errors:
            print(f"오류: {error}", file=sys.stderr)
        return 1
    print("manifest 구조와 물체 단위 분할 누수 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
