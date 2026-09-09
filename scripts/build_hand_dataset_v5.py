#!/usr/bin/env python
"""학습 데이터셋 v5 구성 (서버 실행용): v4의 결함 수정 + Copy-Paste 합성 병합.

실험 N(v4)의 실전 실패 원인을 반영한다:
1. **handle 라벨 누락 프레임 제거**: v4의 머그 동반 손 프레임 중 의사라벨에 handle이
   없는 프레임(20장)은 "보이는 손잡이=배경"이라는 거짓 음성을 가르쳤다 → train에서 제외.
2. **Copy-Paste 합성 병합**: 손-물체 겹침·문맥 다양성을 합성으로 보충
   (scripts/build_hand_copy_paste.py 산출물, train 2,200 / valid 120).
3. **검증 설계 수정**: valid에 합성 상호작용 프레임을 포함해, 지표가 "손이 물체 곁에
   있는" 배포 상황을 대변하게 한다 (v4의 실패 교훈).

test는 v4(=v3) 그대로 둔다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv4"))
    parser.add_argument("--synth-dir", type=Path, default=Path("data/interim/hand_copy_paste"))
    parser.add_argument("--mug-labels", type=Path, default=Path("data/interim/hand_mug_pseudolabels/labels.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/rfdetr_mixed_grasp_type_customv5"))
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (REPOSITORY_ROOT / path)

    v4 = resolve(args.v4_dir)
    synth = resolve(args.synth_dir)
    out = resolve(args.output_dir)

    # handle 라벨이 없는 머그 동반 프레임 = 거짓 음성 유발 → 제외 대상
    bad_frames = set()
    for line in resolve(args.mug_labels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not any(i["class"] == "handle_grasp_region" for i in row["instances"]):
            bad_frames.add(row["image"])
    print(f"제외 대상(handle 라벨 누락): {len(bad_frames)}장")

    stats = {}
    for split in ("train", "valid", "test"):
        coco = json.loads((v4 / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)

        removed_images = 0
        if split == "train" and bad_frames:
            keep_images = [im for im in coco["images"] if im["file_name"] not in bad_frames]
            removed_images = len(coco["images"]) - len(keep_images)
            keep_ids = {im["id"] for im in keep_images}
            coco["images"] = keep_images
            before = len(coco["annotations"])
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
            print(f"train: 이미지 {removed_images}장, 주석 {before - len(coco['annotations'])}개 제거")

        for image_entry in coco["images"]:
            link_or_copy(v4 / split / image_entry["file_name"], split_out / image_entry["file_name"])

        added = 0
        fragment_path = synth / split / "coco_fragment.json"
        if fragment_path.is_file():
            fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
            image_id_offset = max((im["id"] for im in coco["images"]), default=0)
            annotation_id_offset = max((a["id"] for a in coco["annotations"]), default=0)
            id_map = {}
            for image_entry in fragment["images"]:
                new_id = image_entry["id"] + image_id_offset
                id_map[image_entry["id"]] = new_id
                entry = dict(image_entry)
                entry["id"] = new_id
                coco["images"].append(entry)
                link_or_copy(synth / split / image_entry["file_name"],
                             split_out / image_entry["file_name"])
                added += 1
            for annotation in fragment["annotations"]:
                entry = dict(annotation)
                entry["id"] = annotation["id"] + annotation_id_offset
                entry["image_id"] = id_map[annotation["image_id"]]
                coco["annotations"].append(entry)

        (split_out / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        stats[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
                        "removed": removed_images, "synthetic_added": added}
        print(f"{split}: 이미지 {len(coco['images'])} (합성 +{added}, 제거 -{removed_images}), "
              f"주석 {len(coco['annotations'])}")

    manifest = {
        "schema_version": 1,
        "base_dataset": str(args.v4_dir),
        "changes": {
            "removed_handle_missing_mug_frames": sorted(bad_frames),
            "copy_paste_synthetics": "scripts/build_hand_copy_paste.py (seed 42, train 2200/valid 120)",
            "valid_includes_interaction": True,
        },
        "reason": "실험 N 실전 실패 대응: 거짓 음성 제거 + 손-물체 겹침 합성 + 검증 설계 수정",
        "stats": stats,
    }
    (out / "rfdetr_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest 저장 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
