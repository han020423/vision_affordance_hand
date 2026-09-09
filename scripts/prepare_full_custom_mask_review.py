"""최종 자체 머그 140장을 모두 수정할 수 있는 작업 복사본을 준비한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_review import prepare_full_review_workspace  # noqa: E402


def parse_args() -> argparse.Namespace:
    """최종 확인 세트와 새 작업 복사본 경로를 정의한다."""

    parser = argparse.ArgumentParser(
        description="최종 자체 머그 140장 전체를 수정 가능한 복사본으로 준비합니다."
    )
    parser.add_argument(
        "--final-review-root",
        type=Path,
        default=Path("outputs/custom_final_review/final_140_v1"),
        help="현재 최종 140장 확인 세트",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/custom_mask_review/full_140_round2"),
        help="전체 마스크를 수정할 새 작업 폴더",
    )
    return parser.parse_args()


def main() -> int:
    """상대경로를 저장소 기준으로 바꾸고 전체 작업 공간을 생성한다."""

    args = parse_args()
    resolve = lambda path: path if path.is_absolute() else REPOSITORY_ROOT / path
    summary = prepare_full_review_workspace(
        resolve(args.final_review_root), resolve(args.output_root)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
