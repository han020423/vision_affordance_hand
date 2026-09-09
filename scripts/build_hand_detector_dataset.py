#!/usr/bin/env python
"""분리형 손 검출기(1클래스, 박스)용 YOLO detect 데이터셋을 만든다.

실험 N(분할 모델에 robot_hand 통합)의 실전 실패 이후 채택한 방법 ③(분리형 검출기)의
데이터 단계다. 물체 모델(M)과 그 데이터는 일절 건드리지 않고, v5 데이터셋의 COCO
주석에서 robot_hand 박스만 뽑아 별도 검출 데이터셋을 구성한다.

구성:
- 양성: v5 train/valid에서 robot_hand 주석이 있는 이미지 전부
  (실제 젯슨 프레임 96/39장 + Copy-Paste 합성 2,200/120장. 합성은 손-물체 겹침
  장면을 포함하므로 상호작용 강건성에 기여한다.)
- 음성(빈 라벨): 손이 없는 이미지 — custom 촬영본 전부(replay 중복 제거) +
  공개 이미지 표본. 배포 배경에서의 오검출 억제가 목적이다.

원본(v5)은 읽기 전용으로 하드링크(실패 시 심볼릭 링크)만 생성한다.
분할은 v5의 train/valid를 그대로 승계한다(영상 단위·crop 누수 방지가 이미 반영됨).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPLAY_SUFFIX = re.compile(r"__r(?:ep)?\d+$")


def link_image(source: Path, destination: Path) -> None:
    """하드링크를 우선 시도하고 실패하면 심볼릭 링크를 만든다."""

    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source.resolve())


def base_stem(stem: str) -> str:
    """replay 복제 접미사(__rNN)를 제거한 기준 이름을 돌려준다."""

    return REPLAY_SUFFIX.sub("", stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dir", type=Path, required=True,
                        help="rfdetr_mixed_grasp_type_customv5 경로 (읽기 전용)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-neg-train", type=int, default=800)
    parser.add_argument("--public-neg-valid", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest = {"source": str(args.v5_dir), "seed": args.seed, "splits": {}}

    for split in ("train", "valid"):
        split_dir = args.v5_dir / split
        document = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in document["categories"]}
        hand_ids = {i for i, n in categories.items() if n == "robot_hand"}
        if not hand_ids:
            raise SystemExit(f"{split}: robot_hand 카테고리가 없습니다: {categories}")
        images = {i["id"]: i for i in document["images"]}
        hand_boxes: dict[int, list[list[float]]] = {}
        for annotation in document["annotations"]:
            if annotation["category_id"] in hand_ids:
                hand_boxes.setdefault(annotation["image_id"], []).append(annotation["bbox"])

        image_out = args.output / "images" / split
        label_out = args.output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        counts: Counter[str] = Counter()
        # ── 양성: robot_hand 주석 이미지 전부 ──
        for image_id, boxes in sorted(hand_boxes.items()):
            info = images[image_id]
            source = split_dir / info["file_name"]
            if not source.is_file():
                raise SystemExit(f"이미지가 없습니다: {source}")
            width, height = float(info["width"]), float(info["height"])
            lines = []
            for x, y, w, h in boxes:
                cx, cy = (x + w / 2.0) / width, (y + h / 2.0) / height
                nw, nh = w / width, h / height
                if not (0 < nw <= 1 and 0 < nh <= 1):
                    raise SystemExit(f"비정상 박스 {info['file_name']}: {(x, y, w, h)}")
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            link_image(source, image_out / info["file_name"])
            (label_out / f"{Path(info['file_name']).stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            counts["pos_synth" if info["file_name"].startswith("synth_") else "pos_real"] += 1

        # ── 음성: 손 없는 이미지 (custom 전부 + 공개 표본, replay 중복 제거) ──
        negative_ids = [i for i in images if i not in hand_boxes]
        seen_bases: set[str] = set()
        custom_negatives, public_negatives = [], []
        for image_id in sorted(negative_ids):
            name = images[image_id]["file_name"]
            base = base_stem(Path(name).stem)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            (custom_negatives if name.startswith("custom") else public_negatives).append(image_id)
        public_quota = args.public_neg_train if split == "train" else args.public_neg_valid
        rng.shuffle(public_negatives)
        selected = custom_negatives + public_negatives[:public_quota]
        for image_id in selected:
            info = images[image_id]
            link_image(split_dir / info["file_name"], image_out / info["file_name"])
            (label_out / f"{Path(info['file_name']).stem}.txt").write_text("", encoding="utf-8")
            counts["neg_custom" if info["file_name"].startswith("custom") else "neg_public"] += 1

        manifest["splits"][split] = dict(counts)
        print(f"{split}: {dict(counts)}")

    (args.output / "dataset.yaml").write_text(
        "# 분리형 손 검출기 v1 (1클래스 detect)\n"
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "names:\n  0: robot_hand\n", encoding="utf-8")
    (args.output / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
