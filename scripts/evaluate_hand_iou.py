#!/usr/bin/env python
"""robot_hand 클래스의 픽셀 IoU를 측정한다 (젯슨 실행용).

v4 valid의 손 프레임(영상 212955, 승인 39장)에 대해 실험 N 모델의 robot_hand
예측 합집합 마스크와 사람 검수 GT 마스크의 픽셀 IoU를 계산한다.
evaluate_pixel_iou.py의 고정 신뢰도 프로토콜(기본 conf 0.25)을 따른다.

주의: GT는 SAM2 의사라벨을 사람이 승인한 것이므로 완전 수동 GT보다 관대한 기준이다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True, help="승인된 손 마스크 PNG 폴더")
    parser.add_argument("--review", type=Path, required=True, help="review.jsonl (approved만 사용)")
    parser.add_argument("--video", default="hand_20260824_212955", help="valid 분할에 해당하는 영상 이름")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import rfdetr

    model = getattr(rfdetr, args.variant)(pretrain_weights=args.model, device=args.device)

    approved = []
    for line in args.review.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["verdict"] == "approved" and row["image"].startswith(args.video + "/"):
                approved.append(row["image"].split("/", 1)[1])
    approved.sort()
    print(f"평가 프레임 {len(approved)}장 (영상 {args.video}, conf>={args.conf})")

    rows = []
    for name in approved:
        frame = cv2.imread(str(args.frames_dir / name))
        gt = cv2.imread(str(args.masks_dir / Path(name).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        if frame is None or gt is None:
            raise FileNotFoundError(name)
        gt = gt > 127

        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        prediction = np.zeros(frame.shape[:2], dtype=bool)
        if detections is not None and len(detections) > 0 and detections.mask is not None:
            class_ids = np.asarray(detections.class_id, dtype=int)
            hand_id = 4 if class_ids.max() >= 4 else 3   # 카테고리 4(robot_hand), base 자동
            for k in range(len(detections)):
                if int(class_ids[k]) != hand_id:
                    continue
                mask = detections.mask[k].astype(np.uint8)
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                prediction |= mask.astype(bool)

        union = int((prediction | gt).sum())
        intersection = int((prediction & gt).sum())
        iou = intersection / union if union > 0 else None
        rows.append({"image": name, "iou": round(iou, 4) if iou is not None else None,
                     "gt_px": int(gt.sum()), "pred_px": int(prediction.sum())})

    ious = [r["iou"] for r in rows if r["iou"] is not None]
    detected = sum(1 for r in rows if r["pred_px"] > 0)
    summary = {
        "frames": len(rows),
        "detected_frames": detected,
        "iou_mean": round(statistics.mean(ious), 4) if ious else None,
        "iou_median": round(statistics.median(ious), 4) if ious else None,
        "iou_min": round(min(ious), 4) if ious else None,
        "conf": args.conf,
        "note": "GT는 SAM2 의사라벨의 사람 승인본",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_image": rows},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
