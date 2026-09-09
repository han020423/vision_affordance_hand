#!/usr/bin/env python
"""손 촬영 프레임 중 머그가 나오는 장면에 머그 의사라벨을 생성한다 (젯슨용).

robot_hand 학습 프레임에서 머그를 배경으로 두면 "머그를 검출하지 마라"는 잘못된
신호가 되므로, 현재 RF-DETR 모델(v3)의 예측을 머그 의사라벨로 기록한다.
모델 자신의 예측을 라벨로 쓰는 자기지도 방식이므로 오버레이 검수를 거친다.

출력 (output-dir 아래):
  masks/<프레임이름>/<k>_<클래스>.png   인스턴스별 0/255 마스크
  overlays/<프레임이름>.jpg             검수용 오버레이
  labels.jsonl                          프레임별 인스턴스 기록
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}
COLORS = {0: (80, 200, 80), 1: (230, 160, 60), 2: (60, 60, 230)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="RF-DETR checkpoint 경로")
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-list", type=Path, required=True, help="처리할 프레임 이름 목록(txt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import rfdetr

    model = getattr(rfdetr, args.variant)(pretrain_weights=args.model, device=args.device)
    names = [n.strip() for n in args.frame_list.read_text(encoding="utf-8").splitlines() if n.strip()]
    print(f"프레임 {len(names)}장, conf>={args.conf}")

    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    records = []

    for i, name in enumerate(names, 1):
        path = args.frames_dir / name
        frame = cv2.imread(str(path))
        if frame is None:
            records.append({"image": name, "status": "read_failed", "instances": []})
            continue
        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)

        instances = []
        overlay = frame.copy()
        if detections is not None and len(detections) > 0 and detections.mask is not None:
            class_ids = np.asarray(detections.class_id, dtype=int)
            base = 1 if class_ids.max() >= len(CLASSES) else 0
            frame_dir = args.output_dir / "masks" / Path(name).stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            for k in range(len(detections)):
                class_id = int(class_ids[k]) - base
                if class_id not in CLASSES:
                    continue
                mask = detections.mask[k].astype(np.uint8)
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                pixels = int(mask.sum())
                if pixels < 200:
                    continue
                confidence = float(detections.confidence[k])
                mask_name = f"{len(instances)}_{CLASSES[class_id]}.png"
                cv2.imwrite(str(frame_dir / mask_name), mask * 255)
                instances.append({"mask": f"{Path(name).stem}/{mask_name}",
                                  "class": CLASSES[class_id],
                                  "confidence": round(confidence, 3), "pixels": pixels})
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, COLORS[class_id], 2)
                if contours:
                    x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.putText(overlay, f"{CLASSES[class_id][:6]} {confidence:.2f}",
                                (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                COLORS[class_id], 2)
        cv2.imwrite(str(args.output_dir / "overlays" / name), overlay)
        records.append({"image": name, "status": "ok", "instances": instances})
        if i % 20 == 0:
            print(f"  {i}/{len(names)}")

    with (args.output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = sum(len(r["instances"]) for r in records)
    empty = sum(1 for r in records if not r["instances"])
    print(f"완료: 인스턴스 {total}개, 검출 없는 프레임 {empty}장. 출력: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
