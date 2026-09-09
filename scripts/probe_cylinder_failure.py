#!/usr/bin/env python
"""원통→handle 실물 재발의 조건을 격리한다 (젯슨 실행용).

오프라인 승인 프레임에서는 P가 원통을 body로 예측하는데(2026-08-26 평가:
미검출 0/286, handle FP 6/286) 실물 시험에서는 handle로 뜬다는 보고에 대해,
촬영-시험의 두 가지 조건 차이를 각각 재현해 어느 쪽이 원인인지 가른다.

  probe scale: 물체가 작게 보이는 상황(더 먼 거리) — 프레임 내용을 축소해
               흰 바탕 캔버스 중앙에 배치 (0.75/0.5/0.35배)
  probe hand:  로봇손 동반 상황 — 승인 손 조각을 물체 옆(겹침 포함)에 합성

각 조건에서 handle/body 예측 프레임 수와 최고 신뢰도 클래스를 집계한다.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

CYLINDER_VIDEOS = ["hand_20260825_194539_cylinders", "hand_20260825_194726_cylinders"]
MIN_PX = 400


def load_approved_frames(labels_root: Path, review_path: Path, limit: int, seed: int):
    review = {}
    for line in review_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            review[row["image"]] = row["verdict"]
    frames = [key for key in sorted(review) if review[key] == "approved"
              and any(key.startswith(v + "/") for v in CYLINDER_VIDEOS)]
    rng = random.Random(seed)
    rng.shuffle(frames)
    return frames[:limit]


def load_hand_crops(frames_dir: Path, masks_dir: Path, review_path: Path, limit: int, seed: int):
    crops = []
    review_rows = [json.loads(l) for l in review_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(review_rows)
    for row in review_rows:
        if row["verdict"] != "approved" or not row["image"].startswith("hand_20260825_194112_approach_hand/"):
            continue
        frame = cv2.imread(str(frames_dir / row["image"]))
        mask = cv2.imread(str(masks_dir / Path(row["image"]).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue
        mask = (mask > 127).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) < 3000:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crops.append((frame[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()))
        if len(crops) >= limit:
            break
    return crops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--frames-root", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--review", type=Path, default=Path("data/interim/v6_object_pseudolabels/review.jsonl"))
    parser.add_argument("--hand-masks", type=Path, default=Path("data/interim/v6_hand_pseudolabels/masks"))
    parser.add_argument("--hand-review", type=Path, default=Path("data/interim/v6_hand_pseudolabels/review.jsonl"))
    parser.add_argument("--labels-root", type=Path, default=Path("data/interim/v6_object_pseudolabels"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-examples", type=Path, default=None,
                        help="조건별 오버레이 예시를 저장할 디렉터리")
    args = parser.parse_args()

    import rfdetr

    model = getattr(rfdetr, args.variant)(pretrain_weights=args.model, device=args.device)
    names = load_approved_frames(args.labels_root, args.review, args.frames, seed=42)
    crops = load_hand_crops(args.frames_root, args.hand_masks, args.hand_review, limit=10, seed=42)
    print(f"프레임 {len(names)}장, 손 조각 {len(crops)}개")

    def classify(frame):
        """(handle_px, body_px, 최고신뢰 클래스) — 0기준 고정."""

        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        handle_px = body_px = 0
        best = (None, 0.0)
        if detections is None or len(detections) == 0 or detections.mask is None:
            return 0, 0, None
        ids = np.asarray(detections.class_id, dtype=int)
        for k in range(len(detections)):
            cid = int(ids[k])
            mask = detections.mask[k].astype(np.uint8)
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            px = int(mask.sum())
            conf = float(detections.confidence[k])
            if cid == 0:
                handle_px = max(handle_px, px)
            elif cid == 1:
                body_px = max(body_px, px)
            if cid in (0, 1) and px >= MIN_PX and conf > best[1]:
                best = ({0: "handle", 1: "body"}[cid], conf)
        return handle_px, body_px, best[0]

    def shrink(frame, factor):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
        # 흰 매트 톤의 캔버스 중앙 배치 (배경 급변을 피하려 프레임 평균색 사용)
        canvas = np.full_like(frame, 0)
        canvas[:] = frame.mean(axis=(0, 1)).astype(np.uint8)
        y0 = (h - small.shape[0]) // 2
        x0 = (w - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return canvas

    def paste_hand(frame, crop, alpha, rng):
        out = frame.copy()
        h, w = out.shape[:2]
        target_w = int(w * rng.uniform(0.22, 0.35))
        scale = target_w / crop.shape[1]
        size = (target_w, max(8, int(crop.shape[0] * scale)))
        c = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        a = cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # 물체는 대략 중앙에 있으므로 중앙 오른쪽에 겹치게 배치
        x = int(w * rng.uniform(0.45, 0.62))
        y = int(h * rng.uniform(0.30, 0.50))
        x2, y2 = min(w, x + c.shape[1]), min(h, y + c.shape[0])
        soft = cv2.GaussianBlur(a[:y2 - y, :x2 - x], (5, 5), 0)[..., None]
        out[y:y2, x:x2] = (out[y:y2, x:x2] * (1 - soft) + c[:y2 - y, :x2 - x] * soft).astype(np.uint8)
        return out

    conditions = ["original", "scale_075", "scale_050", "scale_035", "with_hand"]
    tallies = {c: {"handle_frames": 0, "body_frames": 0, "best_handle": 0, "best_body": 0,
                   "no_detection": 0} for c in conditions}
    rng = random.Random(42)
    examples_saved = {c: 0 for c in conditions}
    if args.save_examples:
        args.save_examples.mkdir(parents=True, exist_ok=True)

    for index, relative in enumerate(names):
        frame = cv2.imread(str(args.frames_root / relative))
        if frame is None:
            continue
        variants = {
            "original": frame,
            "scale_075": shrink(frame, 0.75),
            "scale_050": shrink(frame, 0.50),
            "scale_035": shrink(frame, 0.35),
            "with_hand": paste_hand(frame, *crops[index % len(crops)], rng) if crops else frame,
        }
        for condition, image in variants.items():
            handle_px, body_px, best = classify(image)
            t = tallies[condition]
            if handle_px >= MIN_PX:
                t["handle_frames"] += 1
            if body_px >= MIN_PX:
                t["body_frames"] += 1
            if best == "handle":
                t["best_handle"] += 1
            elif best == "body":
                t["best_body"] += 1
            else:
                t["no_detection"] += 1
            if (args.save_examples and best == "handle" and examples_saved[condition] < 3):
                cv2.imwrite(str(args.save_examples / f"{condition}_{index}.jpg"), image,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                examples_saved[condition] += 1

    result = {"model": args.model, "frames": len(names), "conf": args.conf,
              "note": "best_*는 프레임의 최고 신뢰 grasp 클래스", "tallies": tallies}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["tallies"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
