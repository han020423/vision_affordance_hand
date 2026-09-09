#!/usr/bin/env python
"""학습 데이터셋 v6 구성 (서버 실행용, 2단계).

실험 O의 실전 한계(접근 자세 손 미검출, 원통→handle 퇴행, 검정 물체→robot_hand)를
2026-08-25 신규 촬영 4종의 승인 라벨로 겨냥한다.

구성 (v5를 승계하지 않고 v4에서 다시 쌓는다 — 구 합성본은 개방 손바닥 조각뿐이라
승계하면 편중이 남으므로, 전체 조각 풀로 합성을 새로 만든다):

  stage core:
    v4 − handle 라벨 누락 머그 프레임(v5와 같은 규칙)
    + train: approach_hand 승인 손 마스크(robot_hand=4)
             cylinders×2 승인 물체 라벨(전부 body — 무손잡이 물체)
             black_objects 승인 물체 라벨(handle/body — robot_hand 네거티브 겸용)
    + valid: approach_valid 중 손 검수·물체 검수 **모두 승인**된 프레임
             (robot_hand + 물체 라벨. 한쪽만 승인된 프레임은 라벨 불완전 →
              보이는 무라벨 물체가 거짓 음성이 되므로 제외 — v4 교훈)

  stage final:
    core + build_hand_copy_paste.py 산출 fragment 병합 (v5 빌더와 같은 방식)

test는 v4(=v3) 그대로 둔다. 원본과 승인 마스크는 수정하지 않는다.
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

ROBOT_HAND_CATEGORY = {"id": 4, "name": "robot_hand", "supercategory": "robot"}
CLASS_TO_ID = {"handle_grasp_region": 1, "body_grasp_region": 2, "functional_region": 3}
MIN_COMPONENT_PX = 300

TRAIN_HAND_VIDEO = "hand_20260825_194112_approach_hand"
VALID_INTERACTION_VIDEO = "hand_20260825_194238_approach_valid"
TRAIN_OBJECT_VIDEOS = (
    "hand_20260825_194539_cylinders",
    "hand_20260825_194726_cylinders",
    "hand_20260825_194952_black_objects",
)


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_COMPONENT_PX:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True)
        if len(approx) >= 3:
            polygons.append([float(v) for point in approx.reshape(-1, 2) for v in point])
    return polygons


def mask_annotation(mask: np.ndarray, image_id: int, category_id: int, annotation_id: int):
    polygons = mask_to_polygons(mask)
    if not polygons:
        return None
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return {"id": annotation_id, "image_id": image_id, "category_id": category_id,
            "segmentation": polygons,
            "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            "area": int(mask.sum()), "iscrowd": 0}


def load_review(path: Path) -> dict[str, str]:
    verdicts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            verdicts[row["image"]] = row["verdict"]
    return verdicts


def load_clean_labels(video_dir: Path) -> dict[str, list[dict]]:
    labels = {}
    for line in (video_dir / "labels_clean.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["image"]] = row["instances"]
    return labels


def stage_core(args) -> None:
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path)

    v4 = resolve(args.v4_dir)
    out = resolve(args.output_dir)
    frames = resolve(args.frames_dir)
    hand_dir = resolve(args.hand_labels)
    object_dir = resolve(args.object_labels)

    # v5와 같은 규칙: handle 라벨 누락 머그 동반 프레임은 거짓 음성 → train에서 제외
    bad_frames = set()
    for line in resolve(args.mug_labels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not any(i["class"] == "handle_grasp_region" for i in row["instances"]):
            bad_frames.add(row["image"])
    print(f"제외 대상(handle 라벨 누락): {len(bad_frames)}장")

    hand_review = load_review(hand_dir / "review.jsonl")
    object_review = load_review(object_dir / "review.jsonl")
    object_labels = {video: load_clean_labels(object_dir / video)
                     for video in TRAIN_OBJECT_VIDEOS + (VALID_INTERACTION_VIDEO,)}

    stats = {"train_hand_frames": 0, "train_object_frames": 0, "valid_interaction_frames": 0,
             "hand_annotations": 0, "object_annotations": 0, "skipped_no_polygon": 0}

    def object_annotations_for(video: str, name: str, image_id: int, next_id: int, coco: dict) -> int:
        added = 0
        for instance in object_labels[video].get(name, []):
            category = CLASS_TO_ID.get(instance["class"])
            if category is None:
                continue
            mask = cv2.imread(str(object_dir / video / "masks" / instance["mask"]),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"물체 마스크 없음: {video}/{instance['mask']}")
            annotation = mask_annotation((mask > 127).astype(np.uint8), image_id, category,
                                         next_id + added)
            if annotation is None:
                stats["skipped_no_polygon"] += 1
                continue
            coco["annotations"].append(annotation)
            added += 1
            stats["object_annotations"] += 1
        return added

    for split in ("train", "valid", "test"):
        split_v4 = v4 / split
        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)
        coco = json.loads((split_v4 / "_annotations.coco.json").read_text(encoding="utf-8"))

        if split == "train" and bad_frames:
            keep_images = [im for im in coco["images"] if im["file_name"] not in bad_frames]
            keep_ids = {im["id"] for im in keep_images}
            removed = len(coco["images"]) - len(keep_images)
            coco["images"] = keep_images
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
            print(f"train: handle 누락 프레임 {removed}장 제거")

        for image_entry in coco["images"]:
            link_or_copy(split_v4 / image_entry["file_name"], split_out / image_entry["file_name"])
        if not any(c["id"] == ROBOT_HAND_CATEGORY["id"] for c in coco["categories"]):
            coco["categories"].append(dict(ROBOT_HAND_CATEGORY))

        next_image_id = max((im["id"] for im in coco["images"]), default=0) + 1
        next_annotation_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

        def add_frame(video: str, name: str) -> int:
            nonlocal next_image_id
            frame_path = frames / video / name
            image = cv2.imread(str(frame_path))
            if image is None:
                raise FileNotFoundError(f"프레임 없음: {video}/{name}")
            link_or_copy(frame_path, split_out / name)
            image_id = next_image_id
            next_image_id += 1
            coco["images"].append({"id": image_id, "file_name": name,
                                   "width": image.shape[1], "height": image.shape[0],
                                   "license": 0, "source": video})
            return image_id

        if split == "train":
            # 접근 자세 손 (robot_hand만; 흰 매트 배경이라 다른 물체 없음)
            for relative, verdict in sorted(hand_review.items()):
                video, name = relative.split("/", 1)
                if video != TRAIN_HAND_VIDEO or verdict != "approved":
                    continue
                mask = cv2.imread(str(hand_dir / "masks" / Path(relative).with_suffix(".png")),
                                  cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"손 마스크 없음: {relative}")
                image_id = add_frame(video, name)
                annotation = mask_annotation((mask > 127).astype(np.uint8), image_id,
                                             ROBOT_HAND_CATEGORY["id"], next_annotation_id)
                if annotation is None:
                    stats["skipped_no_polygon"] += 1
                else:
                    coco["annotations"].append(annotation)
                    next_annotation_id += 1
                    stats["hand_annotations"] += 1
                stats["train_hand_frames"] += 1

            # 원통·검정 물체 (물체 라벨만; robot_hand 없음 = 검정 물체 네거티브)
            for video in TRAIN_OBJECT_VIDEOS:
                for relative, verdict in sorted(object_review.items()):
                    if not relative.startswith(video + "/") or verdict != "approved":
                        continue
                    name = relative.split("/", 1)[1]
                    image_id = add_frame(video, name)
                    added = object_annotations_for(video, name, image_id, next_annotation_id, coco)
                    next_annotation_id += added
                    stats["train_object_frames"] += 1

        if split == "valid":
            # 상호작용 검증: 손·물체 검수 모두 승인된 프레임만 (라벨 완전성 보장)
            video = VALID_INTERACTION_VIDEO
            for relative, verdict in sorted(hand_review.items()):
                if not relative.startswith(video + "/") or verdict != "approved":
                    continue
                if object_review.get(relative) != "approved":
                    continue
                name = relative.split("/", 1)[1]
                mask = cv2.imread(str(hand_dir / "masks" / Path(relative).with_suffix(".png")),
                                  cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"손 마스크 없음: {relative}")
                image_id = add_frame(video, name)
                annotation = mask_annotation((mask > 127).astype(np.uint8), image_id,
                                             ROBOT_HAND_CATEGORY["id"], next_annotation_id)
                if annotation is None:
                    stats["skipped_no_polygon"] += 1
                else:
                    coco["annotations"].append(annotation)
                    next_annotation_id += 1
                    stats["hand_annotations"] += 1
                added = object_annotations_for(video, name, image_id, next_annotation_id, coco)
                next_annotation_id += added
                stats["valid_interaction_frames"] += 1

        (split_out / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        print(f"{split}: 이미지 {len(coco['images'])}, 주석 {len(coco['annotations'])}")

    manifest = {
        "schema_version": 1,
        "stage": "core",
        "base_dataset": str(args.v4_dir),
        "changes": {
            "removed_handle_missing_mug_frames": sorted(bad_frames),
            "train_added": {"approach_hand": stats["train_hand_frames"],
                            "objects": stats["train_object_frames"]},
            "valid_added_interaction": stats["valid_interaction_frames"],
        },
        "reason": "실험 O 실전 한계 대응: 접근 자세 손·원통 body·검정 물체 네거티브 실사 추가",
        "stats": stats,
    }
    (out / "rfdetr_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("core 완료:", stats)


def stage_final(args) -> None:
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path)

    core = resolve(args.core_dir)
    synth = resolve(args.synth_dir)
    out = resolve(args.final_dir)

    stats = {}
    for split in ("train", "valid", "test"):
        coco = json.loads((core / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)
        for image_entry in coco["images"]:
            link_or_copy(core / split / image_entry["file_name"],
                         split_out / image_entry["file_name"])

        added = 0
        fragment_path = synth / split / "coco_fragment.json"
        if fragment_path.is_file():
            fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
            image_offset = max((im["id"] for im in coco["images"]), default=0)
            annotation_offset = max((a["id"] for a in coco["annotations"]), default=0)
            id_map = {}
            for image_entry in fragment["images"]:
                new_id = image_entry["id"] + image_offset
                id_map[image_entry["id"]] = new_id
                entry = dict(image_entry)
                entry["id"] = new_id
                coco["images"].append(entry)
                link_or_copy(synth / split / image_entry["file_name"],
                             split_out / image_entry["file_name"])
                added += 1
            for annotation in fragment["annotations"]:
                entry = dict(annotation)
                entry["id"] = annotation["id"] + annotation_offset
                entry["image_id"] = id_map[annotation["image_id"]]
                coco["annotations"].append(entry)

        (split_out / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        stats[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
                        "synthetic_added": added}
        print(f"{split}: 이미지 {len(coco['images'])} (합성 +{added}), 주석 {len(coco['annotations'])}")

    manifest = {
        "schema_version": 1,
        "stage": "final",
        "core_dataset": str(args.core_dir),
        "copy_paste": str(args.synth_dir),
        "reason": "v6 = core(실사) + 전체 조각 풀 Copy-Paste 합성 (구 합성 미승계)",
        "stats": stats,
    }
    (out / "rfdetr_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("final 완료")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["core", "final"], required=True)
    parser.add_argument("--v4-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv4"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--hand-labels", type=Path, default=Path("data/interim/v6_hand_pseudolabels"))
    parser.add_argument("--object-labels", type=Path, default=Path("data/interim/v6_object_pseudolabels"))
    parser.add_argument("--mug-labels", type=Path, default=Path("data/interim/hand_mug_pseudolabels/labels.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv6core"))
    parser.add_argument("--core-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv6core"))
    parser.add_argument("--synth-dir", type=Path, default=Path("data/interim/hand_copy_paste_v6"))
    parser.add_argument("--final-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv6"))
    args = parser.parse_args()

    if args.stage == "core":
        stage_core(args)
    else:
        stage_final(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
