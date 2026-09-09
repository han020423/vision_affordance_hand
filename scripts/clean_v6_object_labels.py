#!/usr/bin/env python
"""v6 물체 의사라벨 정리 (서버 실행용).

M(v3) 모델의 낮은 문턱(0.25) 의사라벨을 검수 전에 규칙으로 정리한다.
2026-08-25 오버레이 판독에서 확인된 계통 오류를 제거하는 것이 목적이다:

1. **고정 배경 오검출**: 벽 콘센트·파워서플라이 전면이 거의 매 프레임 functional로
   잡힌다(카메라 고정이라 위치 불변). 지정 구역 안 중심의 functional을 제거한다.
2. **원통 영상(cylinders)**: 촬영 물체가 전부 무손잡이(종이컵·캔·물병)이므로
   functional은 전부 제거하고 handle은 body로 재지정한다(캔이 handle로 잡히는
   M의 오분류 교정). 사람 손에 잡힌 handle 오검출도 body가 되지만, 그런 프레임은
   검수에서 거부한다.
3. **검정 물체 영상(black_objects)**: 검정 머그·텀블러는 handle/body 분리가 필요
   하므로 클래스는 유지하고 functional만 제거한다.
4. **approach_valid**: SAM2 손 마스크와 지정 비율 이상 겹치는 인스턴스를 제거한다
   (로봇손이 물체 클래스로 이중 검출되는 오염의 라벨 시점 차단 — 런타임
   robot_hand 우선 규칙과 같은 원리).

출력:
  <영상>/labels_clean.jsonl   정리된 라벨 (mask 경로는 원본 masks/ 재사용)
  overlays/<영상>/<프레임>    정리 결과 오버레이 (검수용, 루트에 통합)
  candidates.jsonl            검수 도구(review_hand_masks.py --base) 호환 통합 큐
                              (box_score = 프레임 내 최저 인스턴스 신뢰도)
남은 인스턴스가 0인 프레임은 검수 큐에서 제외한다(학습에도 쓰지 않는다).
원본 labels.jsonl과 masks/는 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = {"handle_grasp_region": 0, "body_grasp_region": 1, "functional_region": 2}
COLORS = {"handle_grasp_region": (80, 200, 80), "body_grasp_region": (230, 160, 60),
          "functional_region": (60, 60, 230)}

# 고정 배경 오검출 구역 (1280x720, 카메라 고정 전제): 파워서플라이 전면 / 벽 콘센트
BACKGROUND_ZONES = [(140, 300, 340, 540), (960, 190, 1120, 320)]


def centroid_in_zones(mask: np.ndarray) -> bool:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False
    cx, cy = float(xs.mean()), float(ys.mean())
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in BACKGROUND_ZONES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path,
                        default=Path("data/interim/v6_object_pseudolabels"))
    parser.add_argument("--frames-root", type=Path,
                        default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--hand-masks-root", type=Path,
                        default=Path("data/interim/v6_hand_pseudolabels/masks"),
                        help="approach_valid 손 겹침 제거에 쓰는 SAM2 마스크")
    parser.add_argument("--hand-overlap-max", type=float, default=0.4,
                        help="인스턴스 면적 중 손 마스크 겹침 비율이 이 이상이면 제거")
    args = parser.parse_args()

    videos = sorted(p.name for p in args.labels_root.iterdir()
                    if p.is_dir() and p.name != "overlays")
    merged_queue = []
    for video in videos:
        video_dir = args.labels_root / video
        rows = [json.loads(line)
                for line in (video_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        is_cylinders = "cylinders" in video
        is_valid = "approach_valid" in video

        overlays_dir = args.labels_root / "overlays" / video
        overlays_dir.mkdir(parents=True, exist_ok=True)
        cleaned = []
        dropped = {"background": 0, "functional": 0, "hand_overlap": 0, "relabeled": 0}

        for row in rows:
            if row["status"] != "ok":
                continue
            frame_path = args.frames_root / video / row["image"]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            hand_mask = None
            if is_valid:
                hand_path = args.hand_masks_root / video / Path(row["image"]).with_suffix(".png")
                if hand_path.is_file():
                    hand = cv2.imread(str(hand_path), cv2.IMREAD_GRAYSCALE)
                    if hand is not None:
                        hand_mask = hand > 127

            kept = []
            overlay = frame.copy()
            for instance in row["instances"]:
                mask_path = video_dir / "masks" / instance["mask"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = mask > 127
                cls = instance["class"]

                # 규칙 1: 고정 배경 구역의 functional 제거 (모든 영상)
                if cls == "functional_region" and centroid_in_zones(mask):
                    dropped["background"] += 1
                    continue
                # 규칙 2·3: 촬영 물체에 없는 클래스 제거
                if cls == "functional_region" and (is_cylinders or "black_objects" in video):
                    dropped["functional"] += 1
                    continue
                # 규칙 4: 손 마스크 겹침 제거 (approach_valid)
                if hand_mask is not None:
                    area = int(mask.sum())
                    if area > 0 and int((mask & hand_mask).sum()) / area >= args.hand_overlap_max:
                        dropped["hand_overlap"] += 1
                        continue
                # 규칙 2: 원통 영상의 handle → body 재지정
                if is_cylinders and cls == "handle_grasp_region":
                    cls = "body_grasp_region"
                    dropped["relabeled"] += 1

                entry = dict(instance)
                entry["class"] = cls
                kept.append(entry)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, COLORS[cls], 2)
                if contours:
                    x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.putText(overlay, f"{cls[:6]} {instance['confidence']:.2f}",
                                (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                COLORS[cls], 2)

            if not kept:
                continue   # 남은 라벨이 없는 프레임은 검수·학습 모두 제외
            cv2.imwrite(str(overlays_dir / row["image"]), overlay)
            cleaned.append({"image": row["image"], "status": "ok", "instances": kept})
            merged_queue.append({"image": f"{video}/{row['image']}", "status": "ok",
                                 "box_score": min(i["confidence"] for i in kept),
                                 "prompt": None, "mask_iou": None,
                                 "mask_px": sum(i["pixels"] for i in kept)})

        with (video_dir / "labels_clean.jsonl").open("w", encoding="utf-8") as handle:
            for row in cleaned:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{video}: 검수 대상 {len(cleaned)}/{len(rows)}장, 제거 {dropped}")

    with (args.labels_root / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged_queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"통합 검수 큐 {len(merged_queue)}장: {args.labels_root / 'candidates.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
