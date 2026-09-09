#!/usr/bin/env python
"""Aff-Grasp 공식 학습 마스크를 프로젝트 통합 라벨 형식으로 변환한다."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.affgrasp import convert_affgrasp  # noqa: E402
from src.datasets.common import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/affgrasp/Data_for_Aff-Grasp/ego_train"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/interim/unified/affgrasp"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/affgrasp.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = convert_affgrasp(
        args.root,
        args.output,
        REPOSITORY_ROOT,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    if not args.dry_run:
        write_jsonl(args.manifest, records)
    statuses = Counter(record.conversion_status for record in records)
    print(f"발견한 Aff-Grasp 샘플: {len(records)}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    if args.dry_run:
        print("시험 실행: 마스크와 manifest를 저장하지 않았습니다.")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
