#!/usr/bin/env python
"""자체 승인 데이터와 공개 데이터를 섞은 replay 미세조정 데이터셋을 생성한다.

목적: 자체 머그 데이터만으로 미세조정하면 공개 데이터에서 배운 다른 범주와
functional_region 지식이 망각된다. 이 스크립트는 원본 두 YOLO 변환본을
수정·복사하지 않고, 이미지 경로 목록(txt)과 dataset.yaml만 생성해
두 데이터를 함께 학습할 수 있게 한다.

- 자체 train 이미지는 --custom-repeat 배수만큼 반복해 공개 데이터 편중을 줄인다.
- Ultralytics는 txt 목록의 각 이미지 경로에서 images→labels 치환으로 라벨을
  찾으므로 라벨 파일을 복사할 필요가 없다.
- train/val 경로 중복(분할 누수), 클래스 매핑 불일치, 누락 파일은 즉시 실패한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# 허용하는 클래스 매핑: 기본 2클래스 또는 파지 종류 3클래스.
# 두 원본 데이터셋은 서로 같은 매핑을 써야 하며, 그 매핑을 출력에 그대로 기록한다.
ALLOWED_NAME_SETS: list[dict[int, str]] = [
    {0: "grasp_region", 1: "functional_region"},
    {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"},
]
EXPECTED_NAMES = ALLOWED_NAME_SETS[0]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: Path) -> str:
    """감사 기록용 대문자 SHA-256 해시를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_source_dataset(root: Path) -> dict[str, Any]:
    """원본 YOLO 데이터셋의 yaml을 읽고 클래스 매핑을 검증한다."""

    yaml_path = root / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml이 없습니다: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    names = {int(key): str(value) for key, value in (document.get("names") or {}).items()}
    if names not in ALLOWED_NAME_SETS:
        raise ValueError(f"클래스 매핑이 허용 정책과 다릅니다({yaml_path}): {names!r}")
    document["names"] = names
    return document


def label_path_for(image_path: Path) -> Path:
    """Ultralytics 규칙과 동일하게 images→labels 치환으로 라벨 경로를 구한다."""

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"경로에 images 디렉터리가 없어 라벨을 찾을 수 없습니다: {image_path}")


def collect_split_images(root: Path, split: str) -> list[Path]:
    """분할 디렉터리의 이미지를 결정적 순서로 나열하고 라벨 존재를 검사한다.

    배경 음성 샘플의 빈 라벨 파일은 정상이므로 내용이 아니라 존재만 확인한다.
    """

    directory = root / "images" / split
    if not directory.is_dir():
        raise FileNotFoundError(f"분할 디렉터리가 없습니다: {directory}")
    images = sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"분할에 이미지가 없습니다: {directory}")
    for image_path in images:
        label = label_path_for(image_path)
        if not label.is_file():
            raise FileNotFoundError(f"라벨 파일이 없습니다: {label} (이미지 {image_path.name})")
    return images


def ensure_no_split_overlap(train_paths: list[Path], val_paths: list[Path]) -> None:
    """train과 val 목록에 같은 이미지가 있으면 분할 누수로 보고 실패한다."""

    leaked = sorted(set(train_paths) & set(val_paths))
    if leaked:
        raise ValueError(
            f"train과 val에 같은 이미지가 있습니다(누수 {len(leaked)}건): {leaked[:3]}"
        )


def build_mixed_dataset(
    custom_root: Path,
    public_root: Path,
    output_root: Path,
    custom_repeat: int,
    order_seed: int | None = None,
) -> dict[str, Any]:
    """혼합 replay 데이터셋(txt 목록 + yaml + manifest)을 생성한다.

    원본 폴더는 읽기 전용으로만 사용하며 출력 폴더가 이미 있으면 기존 생성본을
    덮어쓰지 않도록 실패한다. 반환값은 기록한 manifest 내용이다.
    """

    if custom_repeat < 1:
        raise ValueError(f"custom_repeat는 1 이상이어야 합니다: {custom_repeat}")
    if output_root.exists():
        raise FileExistsError(f"기존 생성본을 덮어쓰지 않습니다: {output_root}")

    custom_document = load_source_dataset(custom_root)
    public_document = load_source_dataset(public_root)
    if custom_document["names"] != public_document["names"]:
        raise ValueError(
            "두 원본 데이터셋의 클래스 매핑이 서로 다릅니다: "
            f"{custom_document['names']!r} != {public_document['names']!r}"
        )
    names = custom_document["names"]

    custom_train = collect_split_images(custom_root, "train")
    custom_val = collect_split_images(custom_root, "val")
    public_train = collect_split_images(public_root, "train")
    public_val = collect_split_images(public_root, "val")

    # 원본 데이터셋 사이에 같은 파일이 섞여 들어오면 분할 누수이므로 즉시 실패한다.
    ensure_no_split_overlap(custom_train + public_train, custom_val + public_val)

    # 자체 데이터를 반복해 공개 데이터 편중을 줄인다. 순서는 결정적으로 유지한다.
    train_entries = [path for _ in range(custom_repeat) for path in custom_train]
    train_entries.extend(public_train)
    if order_seed is not None:
        # seed 분산 측정용: Ultralytics 8.3.163은 데이터로더 generator에
        # args.seed를 반영하지 않아 학습이 완전 결정적이다. 대신 train 목록의
        # 순서를 재배열해 실행 간 무작위성을 만든다 (결정적 재배열, seed 기록).
        import random as _random

        _random.Random(order_seed).shuffle(train_entries)
    val_entries = list(custom_val) + list(public_val)

    output_root.mkdir(parents=True)
    train_txt = output_root / "train.txt"
    val_txt = output_root / "val.txt"
    # Ultralytics가 그대로 읽을 수 있도록 절대경로를 POSIX 구분자로 기록한다.
    train_txt.write_text(
        "\n".join(path.as_posix() for path in train_entries) + "\n", encoding="utf-8"
    )
    val_txt.write_text(
        "\n".join(path.as_posix() for path in val_entries) + "\n", encoding="utf-8"
    )

    dataset_yaml = output_root / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": output_root.resolve().as_posix(),
                "train": "train.txt",
                "val": "val.txt",
                "names": names,
            },
            handle,
            allow_unicode=True,
            sort_keys=False,
        )

    def source_version(root: Path) -> dict[str, Any]:
        version_path = root / "dataset_version.json"
        if version_path.is_file():
            return {"path": str(version_path), "sha256": sha256_file(version_path)}
        return {"path": None, "sha256": None}

    manifest = {
        "schema_version": 1,
        "purpose": "자체 머그 미세조정 시 공개 데이터 지식 망각을 줄이기 위한 replay 혼합",
        "sources": {
            "custom": {"root": str(custom_root.resolve()), "dataset_version": source_version(custom_root)},
            "public": {"root": str(public_root.resolve()), "dataset_version": source_version(public_root)},
        },
        "custom_repeat": custom_repeat,
        "order_seed": order_seed,
        "names": {str(key): value for key, value in names.items()},
        "counts": {
            "custom_train_unique": len(custom_train),
            "custom_val": len(custom_val),
            "public_train": len(public_train),
            "public_val": len(public_val),
            "train_entries_with_repeats": len(train_entries),
            "val_entries": len(val_entries),
            "custom_share_of_train": round(len(custom_train) * custom_repeat / len(train_entries), 4),
        },
        "source_datasets_modified": False,
    }
    (output_root / "mixed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custom",
        type=Path,
        default=Path("data/processed/yolo_custom_approved_v1"),
        help="자체 승인 YOLO 변환본 경로",
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=Path("data/processed/yolo_public_contain_background"),
        help="공개 데이터 YOLO 변환본 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yolo_mixed_replay_v1"),
        help="생성할 혼합 데이터셋 경로",
    )
    parser.add_argument(
        "--custom-repeat",
        type=int,
        default=10,
        help="자체 train 이미지 반복 배수(공개 데이터 편중 완화, 초기값 10)",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="train 목록 재배열 seed (seed 분산 측정용, 기본값 None=결정적 순서)",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    manifest = build_mixed_dataset(
        resolve(args.custom), resolve(args.public), resolve(args.output),
        args.custom_repeat, order_seed=args.order_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("혼합 replay 데이터셋 생성과 검증을 마쳤습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
