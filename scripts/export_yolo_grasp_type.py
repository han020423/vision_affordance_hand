#!/usr/bin/env python
"""파지 종류 3클래스 YOLO 데이터셋을 생성한다.

1단계: 기존 `export_public_yolo`로 2클래스 변환본을 새 경로에 생성
2단계: 각 폴리곤의 클래스를 원본 라벨 정보로 3클래스로 재매핑

원본 데이터와 기존 변환본은 수정하지 않는다.

예시 (공개 데이터, contain 배경 정책):
    python scripts/export_yolo_grasp_type.py \
      --manifests data/interim/affgrasp.jsonl data/interim/umd_8k.jsonl \
      --output data/processed/yolo_grasp_type_public_v1 \
      --ignore-export-policy contain_as_background_keep_grasp \
      --affgrasp-repeat 2

예시 (자체 승인 데이터):
    python scripts/export_yolo_grasp_type.py \
      --manifests data/interim/custom_approved_v1/manifest.jsonl \
      --output data/processed/yolo_grasp_type_custom_v1 \
      --affgrasp-repeat 1 \
      --max-secondary-contour-area 512 \
      --max-hole-contour-area 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.yolo_export import export_public_yolo  # noqa: E402
from src.datasets.yolo_export_grasp_type import remap_export_to_grasp_type  # noqa: E402


def filter_train_object_prefixes(manifest: Path, prefixes: list[str]) -> tuple[Path, int]:
    """train 분할에서 지정 접두어 물체 레코드를 제외한 manifest 사본을 만든다.

    라벨 충돌 실험용: 예) UMD의 mug_*/cup_*는 손잡이까지 wrap-grasp(몸통)로
    라벨돼 자체 데이터와 충돌하므로 train에서만 제외한다. validation/test는
    보존해 기존 실험과의 비교 가능성을 유지한다. 원본 manifest는 수정하지 않는다.
    """

    import json

    kept_lines: list[str] = []
    excluded = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record.get("object_id", ""))
            split = str(record.get("split", ""))
            if split in ("train", "pretrain") and any(
                object_id.startswith(prefix) for prefix in prefixes
            ):
                excluded += 1
                continue
            kept_lines.append(line.rstrip("\n"))
    if excluded == 0:
        return manifest, 0
    suffix = "_no_" + "_".join(p.rstrip("_") for p in prefixes)
    filtered = manifest.with_name(manifest.stem + suffix + ".jsonl")
    filtered.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return filtered, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ignore-export-policy",
        default="exclude_sample",
        choices=["exclude_sample", "contain_as_background_keep_grasp"],
    )
    parser.add_argument("--affgrasp-repeat", type=int, default=2)
    parser.add_argument("--max-secondary-contour-area", type=float, default=0.0)
    parser.add_argument("--max-hole-contour-area", type=float, default=0.0)
    parser.add_argument(
        "--allow-ring-slit",
        action="store_true",
        help="고리형 성분(컵 몸통 등)을 절개선 단일 폴리곤으로 내보낸다",
    )
    parser.add_argument(
        "--exclude-train-object-prefixes",
        default=None,
        help="train/pretrain 분할에서 제외할 object_id 접두어 (쉼표 구분, 예: mug_,cup_)",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    manifests = [resolve(path) for path in args.manifests]
    output = resolve(args.output)

    exclusion_report = {}
    if args.exclude_train_object_prefixes:
        prefixes = [p.strip() for p in args.exclude_train_object_prefixes.split(",") if p.strip()]
        filtered_manifests = []
        for manifest in manifests:
            filtered, excluded = filter_train_object_prefixes(manifest, prefixes)
            filtered_manifests.append(filtered)
            if excluded:
                exclusion_report[manifest.name] = excluded
        manifests = filtered_manifests

    # 1단계: 검증된 기존 export 코드로 2클래스 변환본을 생성한다.
    _, export_summary = export_public_yolo(
        manifests,
        output,
        REPOSITORY_ROOT,
        affgrasp_repeat=args.affgrasp_repeat,
        ignore_export_policy=args.ignore_export_policy,
        max_secondary_contour_area=args.max_secondary_contour_area,
        max_hole_contour_area=args.max_hole_contour_area,
        allow_ring_slit=args.allow_ring_slit,
    )

    # 2단계: 원본 라벨 정보로 클래스를 파지 종류 3클래스로 재매핑한다.
    remap_summary = remap_export_to_grasp_type(output, manifests, REPOSITORY_ROOT)

    print(json.dumps(
        {
            "output": str(output),
            "train_exclusions": exclusion_report,
            "split_images": export_summary["split_images"],
            "grasp_type_counts": remap_summary["counts"],
            "grasp_type_class_instances": remap_summary["class_instances"],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    print("파지 종류 3클래스 데이터셋 생성을 마쳤습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
