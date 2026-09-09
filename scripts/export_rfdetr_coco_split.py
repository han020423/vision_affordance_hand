#!/usr/bin/env python
"""YOLO 변환본의 한 분할만 RF-DETR/COCO 평가 폴더로 변환한다.

mug_04 unseen-instance test(자체 v3 test 분할)처럼 학습이 아닌 평가 전용
묶음을 만들 때 쓴다. export_rfdetr_coco.py의 변환 함수를 그대로 재사용하며,
evaluate_pixel_iou.py가 읽는 구조(<출력>/valid/_annotations.coco.json)로 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from export_rfdetr_coco import build_coco_split, list_split_images, load_names, source_version  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="YOLO 변환본 경로")
    parser.add_argument("--split", default="test", help="변환할 분할 이름(train/val/test)")
    parser.add_argument("--output", type=Path, required=True, help="출력 COCO 폴더")
    parser.add_argument(
        "--object-prefix", default=None,
        help="파일명 접두사로 물체를 제한한다(예: custom__mug_04__). 생략하면 분할 전체",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    source = resolve(args.source)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"기존 변환본을 덮어쓰지 않습니다: {output}")
    load_names(source)
    images = list_split_images(source, args.split)
    if args.object_prefix:
        images = [path for path in images if path.name.startswith(args.object_prefix)]
    if not images:
        raise ValueError("변환할 이미지가 없습니다. 분할 이름과 접두사를 확인하세요.")

    counters = {"hardlink": 0, "copy": 0, "negative_images": 0}
    stats = build_coco_split([(path, 1) for path in images], output / "valid", counters)
    manifest = {
        "purpose": f"{args.split} 분할 평가 전용 COCO 변환본(학습 사용 금지)",
        "source": {"root": str(source), "dataset_version": source_version(source)},
        "split": args.split,
        "object_prefix": args.object_prefix,
        "stats": stats,
        "counters": counters,
    }
    (output / "rfdetr_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"images": stats["images"], "annotations": stats["annotations"], "per_class": stats["per_class"]}, ensure_ascii=False, indent=2))
    print(f"평가용 COCO 변환본을 만들었습니다: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
