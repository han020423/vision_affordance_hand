#!/usr/bin/env python
"""Aff-Grasp와 UMD 통합 마스크를 이식 가능한 YOLO 데이터셋으로 내보낸다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.yolo_export import IGNORE_EXPORT_POLICIES, export_public_yolo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[Path("data/interim/affgrasp.jsonl"), Path("data/interim/umd.jsonl")],
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/yolo_public"))
    parser.add_argument("--affgrasp-repeat", type=int, default=2)
    parser.add_argument(
        "--ignore-export-policy",
        choices=sorted(IGNORE_EXPORT_POLICIES),
        default="exclude_sample",
        help=(
            "ignore 처리 방식입니다. 기본값은 샘플 전체 제외이며, "
            "contain_as_background_keep_grasp는 승인된 비교 실험에서만 사용합니다."
        ),
    )
    parser.add_argument(
        "--max-secondary-contour-area",
        type=float,
        default=0.0,
        help=(
            "YOLO 변환본에서 가장 큰 영역과 떨어진 작은 윤곽을 제거할 최대 면적입니다. "
            "0이면 정리하지 않습니다. 원본 승인 마스크는 수정하지 않습니다."
        ),
    )
    parser.add_argument(
        "--max-hole-contour-area",
        type=float,
        default=0.0,
        help=(
            "YOLO 변환본에서 채울 작은 내부 구멍의 최대 면적입니다. "
            "0이면 정리하지 않습니다. 원본 승인 마스크는 수정하지 않습니다."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _, summary = export_public_yolo(
        args.manifests,
        args.output,
        REPOSITORY_ROOT,
        affgrasp_repeat=args.affgrasp_repeat,
        ignore_export_policy=args.ignore_export_policy,
        max_secondary_contour_area=args.max_secondary_contour_area,
        max_hole_contour_area=args.max_hole_contour_area,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("시험 실행: YOLO 데이터셋 파일을 저장하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
