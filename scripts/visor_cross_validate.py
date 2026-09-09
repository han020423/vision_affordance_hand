#!/usr/bin/env python
"""VISOR 사람 판정과 I 모델 자동 판정의 일치율을 측정한다.

사람이 판정한 1층 표본에서 무작위 100건을 뽑아, 모델이 예측한 영역 중
손-물체 접촉 지점과 가장 많이 겹치는 영역의 클래스를 자동 판정으로 삼고
사람 판정과 비교한다. "자동 판정 가능성"을 보이는 부록 근거용이며,
prior 통계 자체는 사람 판정만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# 사람 판정 → 3클래스 대응 (rim_pinch는 집기 계열이므로 handle로 접는다)
HUMAN_TO_CLASS = {
    "handle": "handle_grasp_region",
    "rim_pinch": "handle_grasp_region",
    "body": "body_grasp_region",
    "functional_touch": "functional_region",
}
MODEL_NAMES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}


def rasterize(segments, shape) -> np.ndarray:
    """VISOR 폴리곤 목록을 이진 마스크로 만든다."""

    mask = np.zeros(shape[:2], dtype=np.uint8)
    for segment in segments or []:
        points = np.asarray(segment, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 3:
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--zip-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    samples = {json.loads(l)["image_name"] + "|" + json.loads(l)["hand_name"]: json.loads(l)
               for l in args.samples.read_text(encoding="utf-8").splitlines() if l.strip()}
    judged = [json.loads(l) for l in args.judgments.read_text(encoding="utf-8").splitlines() if l.strip()]
    eligible = [row for row in judged
                if row["tier"] == 1 and row["judgment"] in HUMAN_TO_CLASS]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.limit]

    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / "hjh_vision_hand" / ".ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(args.model))

    agree = 0
    no_prediction = 0
    confusion: Counter = Counter()
    rows = []
    for row in selected:
        key = row["image_name"] + "|" + row["hand_name"]
        event = samples.get(key)
        if event is None:
            continue
        zip_path = args.zip_root / f"{event['video']}.zip"
        with zipfile.ZipFile(zip_path) as handle:
            names = {Path(n).name: n for n in handle.namelist()}
            member = names.get(event["image_name"])
            if member is None:
                continue
            frame = cv2.imdecode(np.frombuffer(handle.read(member), np.uint8), cv2.IMREAD_COLOR)
        hand_mask = rasterize(event["hand_segments"], frame.shape)
        object_mask = rasterize(event["object_segments"], frame.shape)
        # 접촉 지대: 손 마스크를 넓힌 뒤 물체 마스크와 교차
        kernel = np.ones((31, 31), np.uint8)
        contact = cv2.dilate(hand_mask.astype(np.uint8), kernel).astype(bool) & object_mask
        if not contact.any():
            contact = cv2.dilate(hand_mask.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & object_mask

        result = model.predict(frame, imgsz=640, conf=0.25, device=args.device,
                               retina_masks=True, verbose=False)[0]
        best_class = None
        best_overlap = 0
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for mask, class_id in zip(result.masks.data.cpu().numpy(), classes):
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                overlap = int(np.count_nonzero((mask > 0.5) & contact))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_class = MODEL_NAMES.get(int(class_id))
        human_class = HUMAN_TO_CLASS[row["judgment"]]
        if best_class is None:
            no_prediction += 1
            confusion[(human_class, "no_prediction")] += 1
        else:
            confusion[(human_class, best_class)] += 1
            if best_class == human_class:
                agree += 1
        rows.append({"index": row["index"], "category": row["category"],
                     "human": human_class, "model": best_class, "overlap_px": best_overlap})

    evaluated = len(rows)
    predicted = evaluated - no_prediction
    summary = {
        "evaluated": evaluated,
        "model_no_prediction": no_prediction,
        "agreement_over_predicted": round(agree / predicted, 4) if predicted else None,
        "agreement_over_all": round(agree / evaluated, 4) if evaluated else None,
        "confusion": {f"{h}->{m}": c for (h, m), c in sorted(confusion.items())},
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
