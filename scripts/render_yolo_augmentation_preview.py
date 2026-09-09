#!/usr/bin/env python
"""실제 Ultralytics 학습 변환을 적용한 결정적 미리보기 이미지를 생성한다."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/public_pretrain.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/training_previews/public_pretrain_augmentations.jpg"),
    )
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count는 양수여야 합니다")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    with args.config.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    training = dict(document["training"])
    for key in ("data", "model", "project"):
        path = Path(str(training[key]))
        if not path.is_absolute():
            training[key] = str((REPOSITORY_ROOT / path).resolve())
    hyp = get_cfg(overrides=training)
    # 변환된 각 인스턴스가 어떤 클래스인지 구분되도록 겹친 마스크를 합치지 않는다.
    hyp.overlap_mask = False
    data = check_det_dataset(str(training["data"]), autodownload=False)
    dataset = YOLODataset(
        img_path=data["train"],
        data=data,
        task="segment",
        imgsz=int(training["imgsz"]),
        augment=True,
        hyp=hyp,
        cache=False,
        rect=False,
        prefix="preview: ",
    )
    if len(dataset) < args.count:
        raise ValueError(f"이미지 {len(dataset)}장에서 미리보기 {args.count}장을 요청했습니다")
    indices = [round(index * (len(dataset) - 1) / (args.count - 1)) for index in range(args.count)] if args.count > 1 else [len(dataset) // 2]

    tile_width, tile_height = 320, 350
    columns = 4
    rows = (args.count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    colors = {
        0: np.array([40, 210, 90], dtype=np.float32),
        1: np.array([235, 70, 70], dtype=np.float32),
    }
    seed = int(training["seed"])
    for position, dataset_index in enumerate(indices):
        random.seed(seed + position)
        np.random.seed(seed + position)
        torch.manual_seed(seed + position)
        sample = dataset[dataset_index]
        image = sample["img"].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        masks = sample["masks"].float()
        if masks.numel():
            masks = F.interpolate(
                masks[:, None], size=image.shape[:2], mode="nearest"
            )[:, 0].cpu().numpy()
        classes = sample["cls"].view(-1).cpu().numpy().astype(int)
        for mask, class_id in zip(masks, classes):
            selected = mask > 0.5
            image[selected] = 0.5 * image[selected] + 0.5 * colors[class_id]
        preview = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        preview.thumbnail((tile_width, tile_height - 30))
        x = (position % columns) * tile_width
        y = (position // columns) * tile_height
        sheet.paste(preview, (x, y))
        draw.text(
            (x + 4, y + tile_height - 26),
            Path(str(sample["im_file"])).stem[:42],
            fill="black",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
