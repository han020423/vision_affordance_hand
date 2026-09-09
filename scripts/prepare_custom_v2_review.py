#!/usr/bin/env python
"""2026-08-21 추가 촬영(mug_04·가위·드라이버) 검수 작업 공간을 만든다.

반자동 후보 실행(new132_geometric)의 모든 이미지를 수정 가능한 복사본으로
준비한다. 원본 이미지와 후보 폴더는 절대 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_review import prepare_candidate_review_workspace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("data/interim/custom_pseudolabels_v2/new132_geometric"),
        help="반자동 후보 실행 폴더(candidate_manifest.jsonl 포함)",
    )
    parser.add_argument(
        "--raw-image-root",
        type=Path,
        default=Path("data/raw/custom/images"),
        help="원본 촬영 이미지 폴더",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/custom_mask_review/new132_v2"),
        help="생성할 검수 작업 공간 경로",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    summary = prepare_candidate_review_workspace(
        resolve(args.candidate_root), resolve(args.raw_image_root), resolve(args.output)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"검수 작업 공간을 만들었습니다: {resolve(args.output)}")
    print("수정 화면 실행: python scripts/review_custom_masks.py --workspace", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
