#!/usr/bin/env python
"""통합 의미 마스크 하나를 GUI 없이 육안 확인할 수 있게 렌더링한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


COLORS = {
    1: np.array([40, 210, 90], dtype=np.float32),
    2: np.array([235, 70, 70], dtype=np.float32),
    255: np.array([170, 80, 220], dtype=np.float32),
}


def render_overlay(image_path: Path, mask_path: Path, output_path: Path, alpha: float) -> None:
    """파지·기능·무시 픽셀을 서로 다른 색으로 원본 이미지에 겹친다."""

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if image.shape[:2] != mask.shape:
        raise ValueError(f"이미지와 마스크 크기가 다릅니다: 이미지={image.shape[:2]}, 마스크={mask.shape}")
    output = image.copy()
    for value, color in COLORS.items():
        selected = mask == value
        output[selected] = (1.0 - alpha) * output[selected] + alpha * color
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha는 0과 1 사이여야 합니다")
    render_overlay(args.image, args.mask, args.output, args.alpha)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
