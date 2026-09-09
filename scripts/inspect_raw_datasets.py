#!/usr/bin/env python
"""다운로드한 공개 데이터셋의 구조를 변경 없이 읽기 전용으로 점검한다."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _umd_gt_type(path_text: str) -> str:
    path = Path(path_text)
    data = loadmat(path, variable_names=("gt_type",))
    values = np.asarray(data.get("gt_type", [])).reshape(-1)
    return str(values[0]).strip().lower() if values.size == 1 else "missing_or_invalid"


def inspect_umd(root: Path, jobs: int) -> int:
    images = sorted(root.rglob("*_rgb.jpg"))
    depths = sorted(root.rglob("*_depth.png"))
    labels = sorted(root.rglob("*_label.mat"))
    ranked = sorted(root.rglob("*_label_rank.mat"))
    object_count = sum(1 for path in root.iterdir() if path.is_dir())

    def sample_key(path: Path, suffix: str) -> str:
        relative = path.relative_to(root)
        return (relative.parent / relative.name.removesuffix(suffix)).as_posix()

    image_stems = {sample_key(path, "_rgb.jpg") for path in images}
    depth_stems = {sample_key(path, "_depth.png") for path in depths}
    label_stems = {sample_key(path, "_label.mat") for path in labels}
    ranked_stems = {sample_key(path, "_label_rank.mat") for path in ranked}
    paired = image_stems == depth_stems == label_stems == ranked_stems

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        quality = Counter(executor.map(_umd_gt_type, (str(path) for path in labels), chunksize=64))

    print(f"UMD 물체 수: {object_count}")
    print(f"UMD RGB/depth/label/ranked 수: {len(images)}/{len(depths)}/{len(labels)}/{len(ranked)}")
    print(f"UMD 파일명이 모두 대응하는지 여부: {paired}")
    print(f"UMD gt_type 분포: {dict(sorted(quality.items()))}")
    return 0 if paired and set(quality).issubset({"manual", "automatic"}) else 1


def inspect_affgrasp(root: Path) -> int:
    train = root / "ego_train"
    evaluation = root / "Affordance_Evaluation_Dataset"
    images = sorted(train.glob("*-img.jpg"))
    labels = sorted(train.glob("*-label.png"))
    image_ids = {path.name.removesuffix("-img.jpg") for path in images}
    label_ids = {path.name.removesuffix("-label.png") for path in labels}
    eval_images = list((evaluation / "JPEGImages").glob("*.jpg"))
    eval_labels = list((evaluation / "SegmentationClassNpy").glob("*.npy"))
    print(f"Aff-Grasp 학습 이미지/라벨 수: {len(images)}/{len(labels)}")
    print(f"Aff-Grasp 학습 파일명이 모두 대응하는지 여부: {image_ids == label_ids}")
    print(f"Aff-Grasp AED 이미지/라벨 수: {len(eval_images)}/{len(eval_labels)}")
    return 0 if image_ids == label_ids and len(eval_images) == len(eval_labels) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--affgrasp-root",
        type=Path,
        default=Path("data/raw/affgrasp/Data_for_Aff-Grasp"),
    )
    parser.add_argument("--umd-root", type=Path, default=Path("data/raw/umd/tools"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    return max(inspect_affgrasp(args.affgrasp_root), inspect_umd(args.umd_root, args.jobs))


if __name__ == "__main__":
    raise SystemExit(main())
