#!/usr/bin/env python
"""자체+공개 YOLO 세그 데이터를 RF-DETR 학습용 COCO 형식으로 변환한다.

목적: DETR 계열 비교 실험(실험 K, 2026-08-21에 J에서 개명)을 위해 기존
grasp_type YOLO 변환본을 RF-DETR(Roboflow) 학습 관례인 COCO 폴더 구조로 변환한다.

    출력폴더/
      train/_annotations.coco.json + 이미지
      valid/_annotations.coco.json + 이미지
      test/_annotations.coco.json  + 이미지

원칙 (build_mixed_finetune_dataset.py의 혼합 replay 규칙을 그대로 따른다):

- 원본 두 변환본의 파일은 수정하지 않는다. 이미지는 가능하면 하드링크로 연결해
  디스크 사용을 줄이고, 하드링크가 불가능하면 복사한다.
- 자체 train 이미지는 --custom-repeat 배수만큼 반복해 공개 데이터 편중을 줄인다.
  반복본은 파일명에 __rep{n} 접미사를 붙여 COCO image 항목을 구분한다.
- validation은 자체 val + 공개 val을 병합한다(조기 종료가 두 분포를 함께 반영).
- test는 공개 test만 사용한다. 자체 test(mug_04)는 아직 없다.
- train/val 경로 중복(분할 누수), 클래스 매핑 불일치, 좌표 범위 오류, 파일명
  충돌은 즉시 실패한다. 빈 라벨(배경 음성)은 어노테이션 없는 image 항목으로
  보존하고 수를 기록한다.
- COCO category_id는 1부터 시작한다(YOLO class_id + 1). 매핑은 manifest에 기록한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import struct
import sys
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# 실험 K는 파지 종류 3클래스 계보 전용이다. 다른 매핑이 들어오면 즉시 실패한다.
EXPECTED_NAMES: dict[int, str] = {
    0: "handle_grasp_region",
    1: "body_grasp_region",
    2: "functional_region",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# 정규화 좌표 검증 허용 오차. 이 범위를 넘으면 데이터 오류로 실패한다.
COORDINATE_TOLERANCE = 0.01


def sha256_file(path: Path) -> str:
    """파일 전체를 메모리에 올리지 않고 SHA-256 해시를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_image_size(path: Path) -> tuple[int, int]:
    """이미지 헤더만 읽어 (너비, 높이)를 반환한다.

    JPEG과 PNG만 지원한다. 전체 픽셀을 디코딩하지 않아 대량 처리에 빠르다.
    해석에 실패하면 데이터 오류로 예외를 던진다.
    """

    with path.open("rb") as handle:
        head = handle.read(26)
        # PNG: 8바이트 시그니처 + IHDR 청크의 너비/높이
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", head[16:24])
            return int(width), int(height)
        # JPEG: SOF 마커를 찾을 때까지 세그먼트를 건너뛴다.
        if head.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    break
                code = marker[1]
                if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                    continue
                length = struct.unpack(">H", handle.read(2))[0]
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    handle.seek(1, os.SEEK_CUR)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return int(width), int(height)
                handle.seek(length - 2, os.SEEK_CUR)
    raise ValueError(f"이미지 크기를 해석할 수 없습니다: {path}")


def load_names(root: Path) -> dict[int, str]:
    """원본 YOLO 데이터셋 yaml에서 클래스 매핑을 읽고 정책과 대조한다."""

    yaml_path = root / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml이 없습니다: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    names = {int(key): str(value) for key, value in (document.get("names") or {}).items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"클래스 매핑이 실험 K 정책과 다릅니다({yaml_path}): {names!r}")
    return names


def list_split_images(root: Path, split: str) -> list[Path]:
    """분할 폴더의 이미지를 결정적 순서(파일명 정렬)로 나열한다."""

    directory = root / "images" / split
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path_for(image_path: Path) -> Path:
    """YOLO 관례(images→labels, 확장자 .txt)에 따라 라벨 경로를 계산한다."""

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    else:
        raise ValueError(f"경로에 images 폴더가 없습니다: {image_path}")
    return Path(*parts).with_suffix(".txt")


def parse_yolo_polygons(label_path: Path) -> list[tuple[int, list[float]]]:
    """YOLO 세그 라벨 파일을 (클래스, 정규화 폴리곤) 목록으로 해석한다.

    빈 파일(배경 음성)은 빈 목록을 반환한다. 점이 3개 미만이거나 좌표가
    허용 범위를 벗어나면 데이터 오류로 실패한다.
    """

    if not label_path.is_file():
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_path}")
    polygons: list[tuple[int, list[float]]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id not in EXPECTED_NAMES:
            raise ValueError(f"알 수 없는 클래스 {class_id} ({label_path}:{line_no})")
        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            raise ValueError(f"폴리곤 좌표 수가 잘못됐습니다 ({label_path}:{line_no})")
        for value in coordinates:
            if value < -COORDINATE_TOLERANCE or value > 1 + COORDINATE_TOLERANCE:
                raise ValueError(f"정규화 좌표가 범위를 벗어났습니다 ({label_path}:{line_no})")
        clamped = [min(max(value, 0.0), 1.0) for value in coordinates]
        polygons.append((class_id, clamped))
    return polygons


def polygon_area(points: list[float]) -> float:
    """신발끈 공식으로 픽셀 폴리곤 면적(절댓값)을 계산한다."""

    total = 0.0
    count = len(points) // 2
    for index in range(count):
        x1, y1 = points[2 * index], points[2 * index + 1]
        x2, y2 = points[2 * ((index + 1) % count)], points[2 * ((index + 1) % count) + 1]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def link_or_copy(source: Path, target: Path) -> str:
    """같은 볼륨이면 하드링크, 아니면 복사한다. 반환값은 사용한 방식이다."""

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def source_version(root: Path) -> dict[str, Any]:
    """원본 변환본의 dataset_version.json 경로와 해시를 기록용으로 반환한다."""

    version_path = root / "dataset_version.json"
    if not version_path.is_file():
        raise FileNotFoundError(f"dataset_version.json이 없습니다: {version_path}")
    return {"path": str(version_path.resolve()), "sha256": sha256_file(version_path)}


def build_coco_split(
    entries: list[tuple[Path, int]],
    output_dir: Path,
    counters: dict[str, int],
) -> dict[str, Any]:
    """한 분할의 COCO 문서를 만들고 이미지를 출력 폴더에 연결한다.

    entries는 (원본 이미지 경로, 반복 회차) 목록이다. 반복 회차 1은 원본
    파일명을 유지하고, 2 이상은 __rep{n} 접미사를 붙인다.
    """

    output_dir.mkdir(parents=True, exist_ok=False)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    used_names: set[str] = set()
    annotation_id = 1

    for image_id, (source_path, repeat_index) in enumerate(entries, 1):
        if repeat_index == 1:
            file_name = source_path.name
        else:
            file_name = f"{source_path.stem}__rep{repeat_index}{source_path.suffix}"
        if file_name in used_names:
            raise ValueError(f"출력 파일명이 충돌합니다: {file_name}")
        used_names.add(file_name)
        method = link_or_copy(source_path, output_dir / file_name)
        counters[method] += 1

        width, height = read_image_size(source_path)
        images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        polygons = parse_yolo_polygons(label_path_for(source_path))
        if not polygons:
            counters["negative_images"] += 1
        for class_id, normalized in polygons:
            absolute: list[float] = []
            for index in range(0, len(normalized), 2):
                absolute.append(round(normalized[index] * width, 2))
                absolute.append(round(normalized[index + 1] * height, 2))
            xs = absolute[0::2]
            ys = absolute[1::2]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "segmentation": [absolute],
                    "bbox": [round(value, 2) for value in bbox],
                    "area": round(polygon_area(absolute), 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    document = {
        "info": {"description": "vision_hand 실험 K용 RF-DETR COCO 변환본"},
        "licenses": [],
        "categories": [
            {"id": class_id + 1, "name": name, "supercategory": "affordance"}
            for class_id, name in sorted(EXPECTED_NAMES.items())
        ],
        "images": images,
        "annotations": annotations,
    }
    annotation_path = output_dir / "_annotations.coco.json"
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False)
    per_class: dict[str, int] = {name: 0 for name in EXPECTED_NAMES.values()}
    for annotation in annotations:
        per_class[EXPECTED_NAMES[annotation["category_id"] - 1]] += 1
    return {
        "images": len(images),
        "annotations": len(annotations),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custom",
        type=Path,
        default=Path("data/processed/yolo_grasp_type_custom_v2"),
        help="자체 승인 YOLO 변환본 경로",
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=Path("data/processed/yolo_grasp_type_public_v2"),
        help="공개 YOLO 변환본 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rfdetr_mixed_grasp_type_v2"),
        help="생성할 COCO 변환본 경로",
    )
    parser.add_argument(
        "--custom-repeat",
        type=int,
        default=10,
        help="자체 train 이미지 반복 배수(혼합 replay 규칙과 동일, 초기값 10)",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    custom_root = resolve(args.custom)
    public_root = resolve(args.public)
    output_root = resolve(args.output)
    if args.custom_repeat < 1:
        raise ValueError(f"custom-repeat는 1 이상이어야 합니다: {args.custom_repeat}")
    if output_root.exists():
        raise FileExistsError(f"기존 생성본을 덮어쓰지 않습니다: {output_root}")

    load_names(custom_root)
    load_names(public_root)

    custom_train = list_split_images(custom_root, "train")
    custom_val = list_split_images(custom_root, "val")
    public_train = list_split_images(public_root, "train")
    public_val = list_split_images(public_root, "val")
    public_test = list_split_images(public_root, "test")
    if not custom_train or not public_train:
        raise ValueError("train 이미지가 비어 있습니다. 원본 경로를 확인하세요.")

    # 분할 누수 검사: train과 val에 같은 원본 경로가 있으면 즉시 실패한다.
    train_set = {str(path) for path in custom_train + public_train}
    val_set = {str(path) for path in custom_val + public_val}
    overlap = train_set.intersection(val_set)
    if overlap:
        raise ValueError(f"train/val 분할 누수가 있습니다: {sorted(overlap)[:3]} ...")

    # 혼합 규칙: 자체 train을 반복 배수만큼 앞에 두고 공개 train을 잇는다.
    train_entries: list[tuple[Path, int]] = [
        (path, repeat)
        for repeat in range(1, args.custom_repeat + 1)
        for path in custom_train
    ]
    train_entries.extend((path, 1) for path in public_train)
    valid_entries: list[tuple[Path, int]] = [(path, 1) for path in custom_val + public_val]
    test_entries: list[tuple[Path, int]] = [(path, 1) for path in public_test]

    counters = {"hardlink": 0, "copy": 0, "negative_images": 0}
    split_stats = {
        "train": build_coco_split(train_entries, output_root / "train", counters),
        "valid": build_coco_split(valid_entries, output_root / "valid", counters),
        "test": build_coco_split(test_entries, output_root / "test", counters),
    }

    manifest = {
        "purpose": "실험 K: RF-DETR-Seg 미세조정용 혼합 replay COCO 변환본",
        "format": "roboflow_coco(train/valid/test + _annotations.coco.json)",
        "sources": {
            "custom": {"root": str(custom_root), "dataset_version": source_version(custom_root)},
            "public": {"root": str(public_root), "dataset_version": source_version(public_root)},
        },
        "custom_repeat": args.custom_repeat,
        "category_id_mapping": {
            str(class_id): {"coco_category_id": class_id + 1, "name": name}
            for class_id, name in EXPECTED_NAMES.items()
        },
        "counts": {
            "custom_train_unique": len(custom_train),
            "custom_val": len(custom_val),
            "public_train": len(public_train),
            "public_val": len(public_val),
            "public_test": len(public_test),
            "train_images_with_repeats": split_stats["train"]["images"],
            "custom_share_of_train": round(
                len(custom_train) * args.custom_repeat / split_stats["train"]["images"], 4
            ),
            "negative_images_total": counters["negative_images"],
            "image_files_hardlinked": counters["hardlink"],
            "image_files_copied": counters["copy"],
        },
        "split_stats": split_stats,
        "source_datasets_modified": False,
    }
    manifest_path = output_root / "rfdetr_export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print("RF-DETR용 COCO 변환과 검증을 마쳤습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
