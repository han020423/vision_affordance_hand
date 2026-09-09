#!/usr/bin/env python
"""UMD 공식 10-fold 파일에서 검증 가능한 물체 단위 분할을 생성한다."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.splits import build_umd_object_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/umd"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/datasets/umd_object_splits.yaml"),
    )
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--test-fold", type=int, default=0)
    args = parser.parse_args()

    tools_root = args.root / "tools"
    object_ids = {path.name for path in tools_root.iterdir() if path.is_dir()}
    assignments = build_umd_object_splits(
        args.root / "category_10_fold.txt",
        object_ids,
        validation_fold=args.validation_fold,
        test_fold=args.test_fold,
    )
    document = {
        "version": 1,
        "source": {
            "official_fold_file": "data/raw/umd/category_10_fold.txt",
            "test_fold": args.test_fold,
            "validation_fold": args.validation_fold,
            "train_rule": "나머지 모든 실제 물체 ID",
        },
        "splits": {
            split: sorted(
                object_id for object_id, assigned in assignments.items() if assigned == split
            )
            for split in ("train", "validation", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(args.output)

    counts = Counter(assignments.values())
    print(f"저장 완료: {args.output}")
    print(f"물체 수: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
