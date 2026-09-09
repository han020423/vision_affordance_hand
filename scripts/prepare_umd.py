#!/usr/bin/env python
"""사람이 라벨링한 UMD 도구 프레임을 통합 마스크로 변환한다."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.common import write_jsonl  # noqa: E402
from src.datasets.splits import load_object_splits  # noqa: E402
from src.datasets.umd import convert_umd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/umd/tools"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/unified/umd"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/umd.jsonl"))
    parser.add_argument("--splits", type=Path)
    parser.add_argument(
        "--target-manual-frames",
        type=int,
        default=4000,
        help="결정적으로 선택할 수동 정답 프레임 수입니다. 0이면 모든 수동 프레임을 유지합니다.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="UMD gt_type 메타데이터 색인에 사용할 프로세스 수입니다.",
    )
    parser.add_argument(
        "--manual-frame-modulo",
        type=int,
        help="압축파일에 공식 gt_type 필드가 없을 때만 사용할 대체 설정입니다.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_manual_frames < 0:
        raise ValueError("--target-manual-frames는 음수일 수 없습니다")

    def show_progress(phase: str, completed: int, total: int) -> None:
        print(f"[{phase}] {completed}/{total}", flush=True)

    records = convert_umd(
        args.root,
        args.output,
        REPOSITORY_ROOT,
        load_object_splits(args.splits),
        manual_frame_modulo=args.manual_frame_modulo,
        target_manual_frames=args.target_manual_frames or None,
        jobs=args.jobs,
        progress=show_progress,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    if not args.dry_run:
        write_jsonl(args.manifest, records)
    statuses = Counter(record.conversion_status for record in records)
    selections = Counter(record.selection_status for record in records)
    splits = Counter(
        record.split for record in records if record.selection_status == "selected"
    )
    print(f"발견한 UMD 샘플: {len(records)}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    print(f"선택 상태: {dict(sorted(selections.items(), key=lambda item: str(item[0])))}")
    print(f"선택된 분할별 수: {dict(sorted(splits.items()))}")
    if args.dry_run:
        print("시험 실행: 마스크와 manifest를 저장하지 않았습니다.")
    if not args.splits:
        print("물체 분할 파일이 없어 샘플을 unassigned/human_review 상태로 유지합니다.")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
