#!/usr/bin/env python
"""이식 가능한 YOLO 분할 데이터와 감사 manifest를 검증한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    errors: list[str] = []
    class_instances: Counter[int] = Counter()
    split_images: Counter[str] = Counter()

    with (root / "dataset.yaml").open("r", encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)
    # 허용하는 클래스 체계는 두 가지다: 기본 2클래스와 파지 종류 3클래스.
    # 데이터셋 yaml의 names가 둘 중 하나와 정확히 일치해야 하며,
    # 라벨 파일의 클래스 검사도 같은 매핑을 기준으로 한다.
    allowed_name_sets = [
        {0: "grasp_region", 1: "functional_region"},
        {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"},
    ]
    names = {int(key): str(value) for key, value in dataset.get("names", {}).items()}
    if names in allowed_name_sets:
        expected_names = names
    else:
        expected_names = allowed_name_sets[0]
        errors.append(f"클래스 정책 불일치: {names}")

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            # 아직 촬영하지 않은 예약 test는 압축파일에서 빈 폴더가 사라질 수 있다.
            # 이미지·라벨 폴더가 둘 다 없을 때만 0장으로 인정한다.
            if split == "test" and not image_dir.exists() and not label_dir.exists():
                split_images[split] = 0
                continue
            if not image_dir.is_dir():
                errors.append(f"{split}: 이미지 폴더가 없습니다: {image_dir}")
            if not label_dir.is_dir():
                errors.append(f"{split}: 라벨 폴더가 없습니다: {label_dir}")
            split_images[split] = 0
            continue
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            errors.append(
                f"{split}: 이미지와 라벨 파일명이 다릅니다: "
                f"이미지만 있음={sorted(image_stems-label_stems)[:5]}, "
                f"라벨만 있음={sorted(label_stems-image_stems)[:5]}"
            )
        split_images[split] = len(images)
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"이미지를 읽을 수 없습니다({image_path}): {type(exc).__name__}")
        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) < 7 or len(fields) % 2 == 0:
                    errors.append(f"{label_path}:{line_number}: 폴리곤 필드 수가 잘못되었습니다")
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: 숫자가 아닌 필드가 있습니다")
                    continue
                if class_id not in expected_names:
                    errors.append(f"{label_path}:{line_number}: 잘못된 클래스 {class_id}")
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    errors.append(f"{label_path}:{line_number}: 좌표가 [0,1] 범위를 벗어났습니다")
                class_instances[class_id] += 1

    manifest_path = root / "export_manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    exported_paths: set[str] = set()
    object_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("export_status") not in {
            "exported",
            "exported_with_component_exclusions",
            "exported_negative",
        }:
            continue
        if record.get("source_dataset") == "umd":
            object_splits[str(record["object_id"])].add(str(record["output_split"]))
        for exported in record.get("exports", []):
            image_path = str(exported["image_path"])
            if image_path in exported_paths:
                errors.append(f"내보내기 manifest 경로 중복: {image_path}")
            exported_paths.add(image_path)
            if not (root / image_path).is_file():
                errors.append(f"manifest 이미지 파일 누락: {image_path}")
            if not (root / str(exported["label_path"])).is_file():
                errors.append(f"manifest 라벨 파일 누락: {exported['label_path']}")
    for object_id, splits in sorted(object_splits.items()):
        if len(splits) > 1:
            errors.append(f"물체 분할 누수: {object_id} -> {sorted(splits)}")

    actual_image_count = sum(split_images.values())
    if actual_image_count != len(exported_paths):
        errors.append(
            f"manifest와 이미지 수 불일치: manifest={len(exported_paths)}, 파일={actual_image_count}"
        )
    print(f"분할별 이미지 수: {dict(sorted(split_images.items()))}")
    print(f"클래스별 인스턴스 수: {dict(sorted(class_instances.items()))}")
    print(f"감사 레코드 수: {len(records)}")
    if errors:
        for error in errors:
            print(f"오류: {error}", file=sys.stderr)
        return 1
    print("YOLO 데이터 구조, 라벨, 감사 manifest와 물체 분할 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
