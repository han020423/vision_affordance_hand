#!/usr/bin/env python
"""IIT-AFF 라벨을 프로젝트 통합 형식으로 변환한다.

사용자 승인(2026-08-19)으로 추가된 공개 데이터셋이다. 장면 단위 데이터라
물체 ID가 없으므로 전량 pretrain(train 전용)으로 기록하며 validation/test에
사용하지 않는다.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.common import write_jsonl  # noqa: E402
from src.datasets.iit_aff import convert_iit_aff  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/iit_aff/IIT_Affordances_2017"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/interim/unified/iit_aff"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/iit_aff.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = convert_iit_aff(
        args.root,
        args.output,
        REPOSITORY_ROOT,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    if not args.dry_run:
        write_jsonl(args.manifest, records)
    statuses = Counter(record.conversion_status for record in records)
    print(f"발견한 IIT-AFF 샘플: {len(records)}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    if args.dry_run:
        print("시험 실행: 마스크와 manifest를 저장하지 않았습니다.")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
