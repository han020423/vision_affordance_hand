"""전체 수정이 끝난 자체 머그와 배경 사진을 승인 데이터셋으로 승격한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_approval import promote_full_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    """승인 입력과 출력 경로를 정의한다."""

    parser = argparse.ArgumentParser(
        description="전체 검수한 자체 머그 140장과 배경 20장을 승인 데이터로 승격합니다."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/custom_mask_review/full_140_round2"),
        help="전체 140장 수정 작업 폴더",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/interim/custom.jsonl"),
        help="촬영 구간과 고정 분할 정보가 있는 원본 manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/custom_approved_v1"),
        help="기존 중간 결과를 건드리지 않고 만들 승인 데이터 폴더",
    )
    parser.add_argument(
        "--minimum-fragment-pixels",
        type=int,
        default=64,
        help="승격 시 제거할 브러시 잔여 연결 성분 최소 크기",
    )
    return parser.parse_args()


def main() -> int:
    """경로를 저장소 기준으로 해석하고 승인 승격을 실행한다."""

    args = parse_args()
    resolve = lambda path: path if path.is_absolute() else REPOSITORY_ROOT / path
    summary = promote_full_review(
        resolve(args.workspace),
        resolve(args.source_manifest),
        resolve(args.output_root),
        REPOSITORY_ROOT,
        minimum_fragment_pixels=args.minimum_fragment_pixels,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
