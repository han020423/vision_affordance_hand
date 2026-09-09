#!/usr/bin/env python
"""robot_hand 클래스를 추가한 RF-DETR 학습 데이터셋 v4를 구성한다 (서버 실행용).

v3(data/processed/rfdetr_mixed_grasp_type_customv3)를 그대로 물려받고, 승인된
로봇손 프레임을 추가한다. 기존 이미지·라벨은 수정하지 않는다(하드링크).

새 프레임 규칙:
- robot_hand는 category id 4로 **뒤에 추가**한다. 기존 1~3의 의미는 불변(2026-08-24 승인).
- 분할은 영상 단위: train = 212838, 212915 / valid = 212955. test는 v3 그대로(오염 금지).
- 손 마스크는 사람 검수(review.jsonl)에서 approved인 프레임만 쓴다.
- 머그가 나오는 212915 프레임에는 RF-DETR 의사라벨의 handle/body만 병합한다.
  * functional_region 의사라벨은 전부 제외한다: 확인 결과 로봇손·배경 장비의
    오검출이었다(2026-08-24, 9/9건이 손 위에 그려짐).
  * 손 마스크와 겹침(인스턴스 면적 대비)이 0.3 이상인 의사라벨은 손 오검출로 보고 제외한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TRAIN_VIDEOS = {"hand_20260824_212838", "hand_20260824_212915"}
VALID_VIDEOS = {"hand_20260824_212955"}
MUG_VIDEO = "hand_20260824_212915"
ROBOT_HAND_CATEGORY = {"id": 4, "name": "robot_hand", "supercategory": "robot"}
MUG_CLASS_TO_ID = {"handle_grasp_region": 1, "body_grasp_region": 2}
HAND_OVERLAP_DROP = 0.3
MIN_COMPONENT_PX = 300


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """0/1 마스크를 COCO 폴리곤 목록으로 바꾼다 (기존 변환기와 같이 홀 없는 규약)."""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_COMPONENT_PX:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True)
        if len(approx) < 3:
            continue
        polygons.append([float(v) for point in approx.reshape(-1, 2) for v in point])
    return polygons


def mask_annotation(mask: np.ndarray, image_id: int, category_id: int, annotation_id: int):
    polygons = mask_to_polygons(mask)
    if not polygons:
        return None
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygons,
        "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
        "area": int(mask.sum()),
        "iscrowd": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv3"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--hand-labels", type=Path, default=Path("data/interim/hand_pseudolabels"))
    parser.add_argument("--mug-labels", type=Path, default=Path("data/interim/hand_mug_pseudolabels"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv4"))
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path)

    v3_dir = resolve(args.v3_dir)
    frames_dir = resolve(args.frames_dir)
    hand_dir = resolve(args.hand_labels)
    mug_dir = resolve(args.mug_labels)
    out_dir = resolve(args.output_dir)

    review = {}
    for line in (hand_dir / "review.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            review[row["image"]] = row["verdict"]
    approved = sorted(image for image, verdict in review.items() if verdict == "approved")
    print(f"승인된 손 프레임 {len(approved)}장")

    mug_instances: dict[str, list[dict]] = {}
    mug_labels_path = mug_dir / "labels.jsonl"
    if mug_labels_path.is_file():
        for line in mug_labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                mug_instances[row["image"]] = row["instances"]

    stats = {"hand_annotations": 0, "mug_annotations": 0,
             "dropped_functional": 0, "dropped_hand_overlap": 0, "skipped_no_polygon": 0}

    for split in ("train", "valid", "test"):
        split_v3 = v3_dir / split
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        coco = json.loads((split_v3 / "_annotations.coco.json").read_text(encoding="utf-8"))

        for image_entry in coco["images"]:
            link_or_copy(split_v3 / image_entry["file_name"], split_out / image_entry["file_name"])

        if not any(c["id"] == ROBOT_HAND_CATEGORY["id"] for c in coco["categories"]):
            coco["categories"].append(dict(ROBOT_HAND_CATEGORY))

        next_image_id = max((im["id"] for im in coco["images"]), default=0) + 1
        next_annotation_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

        if split == "train":
            videos = TRAIN_VIDEOS
        elif split == "valid":
            videos = VALID_VIDEOS
        else:
            videos = set()

        added_images = 0
        for relative in approved:
            video, name = relative.split("/", 1)
            if video not in videos:
                continue
            mask_path = hand_dir / "masks" / Path(relative).with_suffix(".png")
            hand_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            frame_path = frames_dir / relative
            if hand_mask is None or not frame_path.is_file():
                raise FileNotFoundError(f"마스크 또는 프레임 없음: {relative}")
            hand_mask = (hand_mask > 127).astype(np.uint8)
            height, width = hand_mask.shape

            link_or_copy(frame_path, split_out / name)
            image_id = next_image_id
            next_image_id += 1
            coco["images"].append({"id": image_id, "file_name": name,
                                   "width": width, "height": height,
                                   "license": 0, "source": "hand_jetson_20260824"})
            added_images += 1

            annotation = mask_annotation(hand_mask, image_id, ROBOT_HAND_CATEGORY["id"], next_annotation_id)
            if annotation is None:
                stats["skipped_no_polygon"] += 1
            else:
                coco["annotations"].append(annotation)
                next_annotation_id += 1
                stats["hand_annotations"] += 1

            if video == MUG_VIDEO:
                for instance in mug_instances.get(name, []):
                    if instance["class"] not in MUG_CLASS_TO_ID:
                        stats["dropped_functional"] += 1
                        continue
                    instance_mask = cv2.imread(str(mug_dir / "masks" / instance["mask"]),
                                               cv2.IMREAD_GRAYSCALE)
                    if instance_mask is None:
                        raise FileNotFoundError(f"머그 마스크 없음: {instance['mask']}")
                    instance_mask = (instance_mask > 127).astype(np.uint8)
                    overlap = int((instance_mask & hand_mask).sum()) / max(int(instance_mask.sum()), 1)
                    if overlap >= HAND_OVERLAP_DROP:
                        stats["dropped_hand_overlap"] += 1
                        continue
                    annotation = mask_annotation(instance_mask, image_id,
                                                 MUG_CLASS_TO_ID[instance["class"]],
                                                 next_annotation_id)
                    if annotation is None:
                        stats["skipped_no_polygon"] += 1
                        continue
                    coco["annotations"].append(annotation)
                    next_annotation_id += 1
                    stats["mug_annotations"] += 1

        (split_out / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        print(f"{split}: 이미지 {len(coco['images'])} (신규 {added_images}), "
              f"주석 {len(coco['annotations'])}, 카테고리 {len(coco['categories'])}")

    manifest = {
        "schema_version": 1,
        "base_dataset": str(args.v3_dir),
        "added_source": "hand_jetson_20260824 (승인 155장 중 train 116, valid 39)",
        "robot_hand_category": ROBOT_HAND_CATEGORY,
        "rules": {
            "functional_pseudolabels": "전부 제외 (로봇손·장비 오검출 확인)",
            "hand_overlap_drop": HAND_OVERLAP_DROP,
            "split": {"train": sorted(TRAIN_VIDEOS), "valid": sorted(VALID_VIDEOS)},
        },
        "stats": stats,
    }
    (out_dir / "rfdetr_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("통계:", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
