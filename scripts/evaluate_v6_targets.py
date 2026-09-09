#!/usr/bin/env python
"""v6가 겨냥한 항목의 정량 확인 (젯슨 실행용): 원통=body, 검정 물체≠robot_hand.

승인된 v6 물체 라벨(labels_clean + review approved)을 GT로 쓰고, 지정 모델의
예측을 고정 신뢰도(0.25)로 평가한다. GT는 사람 승인을 거친 의사라벨이므로
완전 수동 GT보다 관대한 기준이다. O/P 두 모델에 같은 프로토콜을 적용해
상대 비교로만 해석한다.

측정:
- cylinders 영상: body 합집합 픽셀 IoU + handle 오검출 프레임 수
  (무손잡이 물체뿐이므로 handle 예측(≥400px)은 전부 오검출)
- black_objects 영상: handle/body 합집합 픽셀 IoU + robot_hand 오검출 프레임 수
  (손이 없는 영상이므로 robot_hand 예측은 전부 오검출; 0.25/0.5 문턱 각각 집계)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

CLASSES = {1: "handle_grasp_region", 2: "body_grasp_region", 3: "functional_region", 4: "robot_hand"}
MIN_FP_PX = 400

CYLINDER_VIDEOS = ["hand_20260825_194539_cylinders", "hand_20260825_194726_cylinders"]
BLACK_VIDEO = "hand_20260825_194952_black_objects"


def load_approved(labels_root: Path, review_path: Path, video: str) -> dict[str, list[dict]]:
    review = {}
    for line in review_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            review[row["image"]] = row["verdict"]
    labels = {}
    for line in (labels_root / video / "labels_clean.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if review.get(f"{video}/{row['image']}") == "approved":
            labels[row["image"]] = row["instances"]
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--frames-root", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--labels-root", type=Path, default=Path("data/interim/v6_object_pseudolabels"))
    parser.add_argument("--review", type=Path, default=Path("data/interim/v6_object_pseudolabels/review.jsonl"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import rfdetr

    model = getattr(rfdetr, args.variant)(pretrain_weights=args.model, device=args.device)

    def predict_unions(frame):
        """클래스별 예측 합집합 마스크 dict를 만든다.

        base는 프레임마다 추정하지 않는다: body만 예측된 프레임에서 min id==1을
        1기준으로 오판해 body→handle로 읽는 버그가 있었다(2026-08-26 확인).
        O/P 런타임은 0기준(0..3, robot_hand=3)임이 실시간 스크립트(min id==0 관측)와
        hand IoU(hand_id=3)에서 확인됐으므로 0으로 고정한다.
        """

        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        unions = {name: np.zeros(frame.shape[:2], dtype=bool) for name in CLASSES.values()}
        if detections is None or len(detections) == 0 or detections.mask is None:
            return unions
        ids = np.asarray(detections.class_id, dtype=int)
        base = 0
        for k in range(len(detections)):
            name = CLASSES.get(int(ids[k]) - base + 1)
            if name is None:
                continue
            mask = detections.mask[k].astype(np.uint8)
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            unions[name] |= mask.astype(bool)
        return unions

    def gt_union(video: str, instances: list[dict], class_name: str, shape) -> np.ndarray:
        union = np.zeros(shape, dtype=bool)
        for instance in instances:
            if instance["class"] != class_name:
                continue
            mask = cv2.imread(str(args.labels_root / video / "masks" / instance["mask"]),
                              cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                union |= mask > 127
        return union

    result = {"model": args.model, "conf": args.conf,
              "note": "GT는 정리+승인 의사라벨(관대한 기준). O/P 상대 비교용."}

    # 원통: body IoU + handle 오검출
    body_ious, handle_fp_frames, frames_total = [], 0, 0
    for video in CYLINDER_VIDEOS:
        for name, instances in sorted(load_approved(args.labels_root, args.review, video).items()):
            frame = cv2.imread(str(args.frames_root / video / name))
            if frame is None:
                continue
            frames_total += 1
            unions = predict_unions(frame)
            gt = gt_union(video, instances, "body_grasp_region", frame.shape[:2])
            union_px = int((unions["body_grasp_region"] | gt).sum())
            if union_px > 0:
                body_ious.append(int((unions["body_grasp_region"] & gt).sum()) / union_px)
            if int(unions["handle_grasp_region"].sum()) >= MIN_FP_PX:
                handle_fp_frames += 1
    result["cylinders"] = {
        "frames": frames_total,
        "body_iou_mean": round(statistics.mean(body_ious), 4) if body_ious else None,
        "body_iou_median": round(statistics.median(body_ious), 4) if body_ious else None,
        "body_complete_miss": sum(1 for v in body_ious if v == 0.0),
        "handle_fp_frames": handle_fp_frames,
    }

    # 검정 물체: handle/body IoU + robot_hand 오검출
    ious = {"handle_grasp_region": [], "body_grasp_region": []}
    hand_fp_025, hand_fp_050, frames_total = 0, 0, 0
    for name, instances in sorted(load_approved(args.labels_root, args.review, BLACK_VIDEO).items()):
        frame = cv2.imread(str(args.frames_root / BLACK_VIDEO / name))
        if frame is None:
            continue
        frames_total += 1
        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        unions = predict_unions(frame)
        for class_name in ious:
            gt = gt_union(BLACK_VIDEO, instances, class_name, frame.shape[:2])
            if not gt.any():
                continue
            union_px = int((unions[class_name] | gt).sum())
            if union_px > 0:
                ious[class_name].append(int((unions[class_name] & gt).sum()) / union_px)
        if int(unions["robot_hand"].sum()) >= MIN_FP_PX:
            hand_fp_025 += 1
            # 0.5 문턱 재판정 (손 추적 채택 문턱과 동일)
            detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=0.5)
            if detections is not None and len(detections) > 0 and detections.mask is not None:
                ids = np.asarray(detections.class_id, dtype=int)
                if any(int(i) + 1 == 4 for i in ids):   # 0기준 고정 (predict_unions 참조)
                    hand_fp_050 += 1
    result["black_objects"] = {
        "frames": frames_total,
        "handle_iou_mean": round(statistics.mean(ious["handle_grasp_region"]), 4)
        if ious["handle_grasp_region"] else None,
        "body_iou_mean": round(statistics.mean(ious["body_grasp_region"]), 4)
        if ious["body_grasp_region"] else None,
        "robot_hand_fp_frames_conf025": hand_fp_025,
        "robot_hand_fp_frames_conf050": hand_fp_050,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
