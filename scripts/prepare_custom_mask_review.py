"""사용자가 표시한 잘못된 반자동 마스크의 수정 작업 공간을 준비한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_review import prepare_review_workspace  # noqa: E402


def parse_args() -> argparse.Namespace:
    """경로를 직접 바꿀 수 있도록 명령행 인자를 정의한다."""

    parser = argparse.ArgumentParser(
        description="사람 검수 의견에 해당하는 마스크만 안전한 수정 폴더로 복사합니다."
    )
    parser.add_argument(
        "--review-config",
        type=Path,
        default=Path("configs/datasets/custom_human_review_20260818.yaml"),
        help="사용자가 작성한 검수 YAML 경로",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "outputs/custom_pseudolabel_review/full_shape_split_v2"
        ),
        help="candidate_manifest.jsonl과 후보 마스크가 있는 폴더",
    )
    parser.add_argument(
        "--raw-image-root",
        type=Path,
        default=Path("data/raw/custom/images"),
        help="자체 촬영 원본 이미지 폴더",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/custom_mask_review/review_20260818"),
        help="원본 후보를 건드리지 않고 생성할 수정 작업 폴더",
    )
    return parser.parse_args()


def main() -> int:
    """경로를 저장소 기준으로 해석하고 작업 공간을 생성한다."""

    args = parse_args()
    resolve = lambda path: path if path.is_absolute() else REPOSITORY_ROOT / path
    summary = prepare_review_workspace(
        resolve(args.review_config),
        resolve(args.candidate_root),
        resolve(args.raw_image_root),
        resolve(args.output_root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
