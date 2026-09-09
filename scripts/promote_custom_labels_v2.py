#!/usr/bin/env python
"""2026-08-21 신규 132장 검수본을 승인 데이터 v2로 승격한다.

승인 v1(160장)은 파일 복사 없이 이어받고(v1 불변), 신규 132장만
카테고리 규칙을 검증해 v2 마스크로 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_approval import promote_new132_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/custom_mask_review/new132_v2"),
        help="사람 검수가 끝난 작업 공간",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/interim/custom_v2.jsonl"),
        help="292장 전체 v2 준비 manifest",
    )
    parser.add_argument(
        "--approved-v1",
        type=Path,
        default=Path("data/interim/custom_approved_v1/manifest.jsonl"),
        help="이어받을 승인 v1 manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/custom_approved_v2"),
        help="생성할 승인 v2 경로",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    summary = promote_new132_review(
        resolve(args.workspace),
        resolve(args.source_manifest),
        resolve(args.approved_v1),
        resolve(args.output),
        REPOSITORY_ROOT,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"승인 v2를 생성했습니다: {resolve(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
