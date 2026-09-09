#!/usr/bin/env python
"""Copy-Paste 합성: 승인된 로봇손 조각을 라벨 완비 물체 이미지에 붙인다 (서버 실행용).

근거: Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for
Instance Segmentation" (CVPR 2021); Dwibedi et al. (ICCV 2017). 실험 N의 실패
(손이 화면에 들어오면 물체 인식 붕괴 — 가림·문맥 변화 미학습)를 촬영 없이
겨냥하는 표준 기법이다.

원리:
- 손 조각: 검수 승인된 robot_hand 마스크로 프레임에서 오려냄 (크기·회전 변형, 좌우반전 없음
  — 실물 손의 좌우가 고정이므로 거울상은 존재하지 않는 데이터다)
- 바탕: v4의 라벨 완비 이미지 (custom 물체 위주 + 일부 public)
- 배치: 일정 확률로 물체 인스턴스와 겹치게 놓아 가림을 학습시킴
- 라벨: 손 = 붙인 마스크 그대로, 물체 = 원 폴리곤에서 손 영역을 뺀 가시 영역
  (30% 미만 남으면 그 인스턴스 라벨 제거) → 거짓 음성이 구조적으로 불가능
- 누수 방지: train 합성은 train 손 조각(영상 212838·212915)·train 바탕만,
  valid 합성은 valid 손 조각(212955)·valid 바탕만 사용

출력: <output-dir>/{train,valid}/ 이미지 + coco_fragment.json (v5 빌더가 병합)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROBOT_HAND_ID = 4
MIN_VISIBLE_PX = 300
MIN_VISIBLE_RATIO = 0.30

# 누수 방지 분할: 2026-08-24 촬영분은 v4 분할 그대로, 2026-08-25 촬영분은
# approach_hand=train / approach_valid=valid (v6 core 분할과 일치).
TRAIN_HAND_VIDEOS = {"hand_20260824_212838", "hand_20260824_212915",
                     "hand_20260825_194112_approach_hand"}
VALID_HAND_VIDEOS = {"hand_20260824_212955", "hand_20260825_194238_approach_valid"}


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_VISIBLE_PX:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True)
        if len(approx) >= 3:
            polygons.append([float(v) for pt in approx.reshape(-1, 2) for v in pt])
    return polygons


def annotation_from_mask(mask: np.ndarray, image_id: int, category_id: int, annotation_id: int):
    polygons = mask_to_polygons(mask)
    if not polygons:
        return None
    ys, xs = np.nonzero(mask)
    return {"id": annotation_id, "image_id": image_id, "category_id": category_id,
            "segmentation": polygons,
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "area": int(mask.sum()), "iscrowd": 0}


def load_hand_crops(frames_dir: Path, masks_dir: Path, review_path: Path, videos: set[str]):
    """승인 프레임에서 (RGB 조각, 알파 마스크) 목록을 만든다."""

    crops = []
    for line in review_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["verdict"] != "approved":
            continue
        video = row["image"].split("/", 1)[0]
        if video not in videos:
            continue
        frame = cv2.imread(str(frames_dir / row["image"]))
        mask = cv2.imread(str(masks_dir / Path(row["image"]).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue
        mask = (mask > 127).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) < 500:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crops.append((frame[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()))
    return crops


def transform_crop(crop, alpha, target_width, angle_deg):
    """조각을 목표 폭으로 스케일하고 회전한다. (반전 없음)"""

    scale = target_width / crop.shape[1]
    new_size = (max(8, int(crop.shape[1] * scale)), max(8, int(crop.shape[0] * scale)))
    crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_NEAREST)
    center = (crop.shape[1] / 2, crop.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
    bound_w = int(crop.shape[0] * sin + crop.shape[1] * cos)
    bound_h = int(crop.shape[0] * cos + crop.shape[1] * sin)
    rotation[0, 2] += bound_w / 2 - center[0]
    rotation[1, 2] += bound_h / 2 - center[1]
    crop = cv2.warpAffine(crop, rotation, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(alpha, rotation, (bound_w, bound_h), flags=cv2.INTER_NEAREST)
    return crop, alpha


def paste(base, crop, alpha, top_left):
    """feather 가장자리로 조각을 합성하고, 전체 크기 손 마스크를 돌려준다."""

    height, width = base.shape[:2]
    x, y = top_left
    ch, cw = crop.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + cw), min(height, y + ch)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_region = crop[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha_region = alpha[y1 - y:y2 - y, x1 - x:x2 - x].astype(np.float32)
    if alpha_region.sum() < 500:
        return None
    soft = cv2.GaussianBlur(alpha_region, (5, 5), 0)[..., None]   # 경계 1~2px feather
    base[y1:y2, x1:x2] = (base[y1:y2, x1:x2] * (1 - soft) + crop_region * soft).astype(np.uint8)
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    hand_mask[y1:y2, x1:x2] = (alpha_region > 0.5).astype(np.uint8)
    return hand_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv4"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--masks-dir", type=Path, action="append", default=None,
                        help="손 마스크 디렉터리 (반복 지정 가능; 미지정 시 hand_pseudolabels/masks)")
    parser.add_argument("--review", type=Path, action="append", default=None,
                        help="검수 파일 (masks-dir와 짝을 이뤄 반복 지정)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/hand_copy_paste"))
    parser.add_argument("--num-train", type=int, default=2200)
    parser.add_argument("--num-valid", type=int, default=120)
    parser.add_argument("--custom-base-ratio", type=float, default=0.7)
    parser.add_argument("--overlap-prob", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path)

    rng = random.Random(args.seed)
    v4 = resolve(args.v4_dir)
    out = resolve(args.output_dir)

    masks_dirs = args.masks_dir or [Path("data/interim/hand_pseudolabels/masks")]
    reviews = args.review or [Path("data/interim/hand_pseudolabels/review.jsonl")]
    if len(masks_dirs) != len(reviews):
        raise SystemExit("--masks-dir와 --review는 같은 횟수로 지정해야 합니다")

    crops_by_split = {"train": [], "valid": []}
    for masks_dir, review in zip(masks_dirs, reviews):
        crops_by_split["train"] += load_hand_crops(
            resolve(args.frames_dir), resolve(masks_dir), resolve(review), TRAIN_HAND_VIDEOS)
        crops_by_split["valid"] += load_hand_crops(
            resolve(args.frames_dir), resolve(masks_dir), resolve(review), VALID_HAND_VIDEOS)
    print(f"손 조각: train {len(crops_by_split['train'])}개, valid {len(crops_by_split['valid'])}개")

    for split, target in (("train", args.num_train), ("valid", args.num_valid)):
        coco = json.loads((v4 / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        annotations_by_image = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        # 바탕 후보: robot_hand 주석이 있는 이미지(실사 손·기존 합성)는 제외.
        # 원통·검정 물체 프레임(hand_2026* 이름이지만 손 없음)은 바탕으로 쓴다 —
        # "접근 자세 손 × 검정 물체" 조합이 배포 시나리오의 핵심이기 때문.
        custom_bases, public_bases = [], []
        for image_entry in coco["images"]:
            name = image_entry["file_name"]
            annotation_list = annotations_by_image.get(image_entry["id"], [])
            if any(a["category_id"] == ROBOT_HAND_ID for a in annotation_list):
                continue
            if not annotation_list:
                continue   # 라벨 없는 이미지는 바탕으로 쓰지 않는다
            is_custom = name.startswith("custom__") or name.startswith("hand_2026")
            (custom_bases if is_custom else public_bases).append(image_entry)
        print(f"{split}: 바탕 후보 custom {len(custom_bases)} / public {len(public_bases)}")

        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)
        images_out, annotations_out = [], []
        next_annotation_id = 1
        made = 0
        attempts = 0
        while made < target and attempts < target * 4:
            attempts += 1
            pool = custom_bases if (rng.random() < args.custom_base_ratio and custom_bases) else public_bases
            if not pool:
                pool = custom_bases or public_bases
            entry = rng.choice(pool)
            base_path = v4 / split / entry["file_name"]
            base = cv2.imread(str(base_path))
            if base is None:
                continue
            height, width = base.shape[:2]
            crop, alpha = rng.choice(crops_by_split[split])
            target_width = int(width * rng.uniform(0.18, 0.40))
            crop_t, alpha_t = transform_crop(crop, alpha, target_width, rng.uniform(-25, 25))

            object_annotations = [a for a in annotations_by_image.get(entry["id"], [])]
            if object_annotations and rng.random() < args.overlap_prob:
                # 물체와 겹치게: 무작위 인스턴스 bbox 가장자리 근처에 손 중심을 둔다
                bx, by, bw, bh = rng.choice(object_annotations)["bbox"]
                cx = bx + bw * rng.uniform(0.0, 1.0) + rng.uniform(-0.3, 0.3) * crop_t.shape[1]
                cy = by + bh * rng.uniform(0.2, 1.1)
            else:
                cx = rng.uniform(0.1, 0.9) * width
                cy = rng.uniform(0.2, 0.95) * height
            top_left = (int(cx - crop_t.shape[1] / 2), int(cy - crop_t.shape[0] / 2))

            composed = base.copy()
            hand_mask = paste(composed, crop_t, alpha_t, top_left)
            if hand_mask is None or hand_mask.sum() < 3000:
                continue

            image_id = made + 1
            file_name = f"synth_{split}_{made:05d}.jpg"
            cv2.imwrite(str(split_out / file_name), composed,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            images_out.append({"id": image_id, "file_name": file_name,
                               "width": width, "height": height, "license": 0,
                               "source": f"copy_paste({entry['file_name']})"})

            hand_annotation = annotation_from_mask(hand_mask, image_id, ROBOT_HAND_ID, next_annotation_id)
            if hand_annotation is None:
                continue
            annotations_out.append(hand_annotation)
            next_annotation_id += 1

            # 물체 라벨: 원 폴리곤 - 손 영역 = 가시 영역
            for annotation in object_annotations:
                instance = np.zeros((height, width), dtype=np.uint8)
                for polygon in annotation["segmentation"]:
                    points = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
                    if len(points) >= 3:
                        cv2.fillPoly(instance, [points], 1)
                original_px = int(instance.sum())
                if original_px == 0:
                    continue
                visible = instance & (1 - hand_mask)
                visible_px = int(visible.sum())
                if visible_px < MIN_VISIBLE_PX or visible_px / original_px < MIN_VISIBLE_RATIO:
                    continue   # 거의 다 가려짐: 라벨 제거 (가시 영역 원칙)
                new_annotation = annotation_from_mask(visible, image_id,
                                                      annotation["category_id"], next_annotation_id)
                if new_annotation is not None:
                    annotations_out.append(new_annotation)
                    next_annotation_id += 1
            made += 1
            if made % 300 == 0:
                print(f"  {split}: {made}/{target}")

        fragment = {"images": images_out, "annotations": annotations_out}
        (split_out / "coco_fragment.json").write_text(json.dumps(fragment), encoding="utf-8")
        print(f"{split} 합성 완료: 이미지 {len(images_out)}, 주석 {len(annotations_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
