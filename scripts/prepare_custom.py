#!/usr/bin/env python
"""자체 촬영 사진을 물체별 라벨링 작업공간과 manifest로 준비한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.custom import prepare_custom_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="원본을 수정하지 않고 자체 촬영 데이터의 라벨링 준비본을 생성합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/custom_captures.yaml"),
        help="촬영 구간, 물체 ID와 분할을 기록한 YAML 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일을 저장하지 않고 구간 배정과 장수만 검증",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    records, summary = prepare_custom_dataset(
        config_path,
        REPOSITORY_ROOT,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("시험 실행: 원본, 작업공간과 manifest를 수정하지 않았습니다.")
    else:
        print(f"자체 데이터 manifest를 생성했습니다: {summary['source_image_count']}장")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())

