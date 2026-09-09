#!/usr/bin/env python
"""녹화 영상으로 후보 선택 파이프라인을 재생·계측한다 (젯슨 실행용).

run_realtime_rfdetr.py의 segmentation 손 소스 경로와 같은 순서로
(검출 → base 고정 → 손 선택 → 픽셀 억제 → 후보 추출 → 억제 필터(유령 포함)
→ 결정 → 파지점 안정화) 프레임을 처리하면서, 파지점이 소실되는 프레임과
그 원인을 로그로 남긴다. "손 근접 시 파지점 소실"(2026-08-26 보고)의
어느 단계가 병목인지 실측하기 위한 도구다. 화면 없이 JSONL만 출력한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_realtime_rfdetr import ROBOT_HAND_CLASS, detections_to_arrays  # noqa: E402
from src.grasp_selection import (  # noqa: E402
    HandState,
    HandSuppressionFilter,
    decide_grasp,
    extract_candidates_from_arrays,
    load_selection_config,
)
from src.perception.hand_tracker import (  # noqa: E402
    HandMotionEstimator,
    SegmentationHandSelector,
    load_hand_tracker_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--video", type=Path, required=True, help="원본 영상(mp4) 또는 프레임 디렉터리")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--hand-seg-conf", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import rfdetr

    model = getattr(rfdetr, args.variant)(pretrain_weights=args.model, device=args.device)
    selection_config = load_selection_config(
        REPOSITORY_ROOT / "configs" / "grasp_selection.yaml",
        REPOSITORY_ROOT / "configs" / "human_grasp_prior.yaml",
    )
    tracker_config = load_hand_tracker_config(
        REPOSITORY_ROOT / "configs" / "hardware" / "hand_tracker.yaml")
    hand_estimator = HandMotionEstimator(
        ema_alpha=tracker_config.direction_ema_alpha,
        min_speed_px=tracker_config.min_speed_px,
        hand_width_mm=tracker_config.hand_width_mm,
        width_px_decay=tracker_config.width_px_decay,
    )
    hand_selector = SegmentationHandSelector(miss_ttl_frames=tracker_config.max_missing_frames)
    suppression_filter = HandSuppressionFilter.from_config(selection_config)

    def frames():
        if args.video.is_dir():
            for path in sorted(args.video.glob("*.jpg")):
                frame = cv2.imread(str(path))
                if frame is not None:
                    yield frame
        else:
            capture = cv2.VideoCapture(str(args.video))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
            capture.release()

    hand_state = None
    hand_missing_streak = 0
    rows = []
    for index, frame in enumerate(frames()):
        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        masks_all, ids_all, confs_all = detections_to_arrays(detections, 0)   # base 0 고정(실측)

        # 손 선택 (실시간 스크립트와 동일)
        hand_select = ids_all == ROBOT_HAND_CLASS
        hand_conf_map = None
        if hand_select.any():
            hand_conf_map = np.zeros(masks_all.shape[1:], dtype=np.float32)
            for hand_mask, hand_conf in zip(masks_all[hand_select], confs_all[hand_select]):
                np.maximum(hand_conf_map, hand_mask * float(hand_conf), out=hand_conf_map)
        track_ids = np.flatnonzero(hand_select & (confs_all >= args.hand_seg_conf))
        hand_infos = []
        for i in track_ids:
            ys, xs = np.nonzero(masks_all[i])
            if len(xs):
                hand_infos.append((int(i), (float(xs.mean()), float(ys.mean())),
                                   (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
        pick = hand_selector.select([h[1] for h in hand_infos],
                                    [float(confs_all[h[0]]) for h in hand_infos])
        hand_box = None
        if pick is not None:
            chosen = hand_infos[pick]
            x1, y1, x2, y2 = chosen[2]
            margin = 0.6 * max(x2 - x1 + 1, y2 - y1 + 1)
            union = masks_all[chosen[0]].copy()
            for info in hand_infos:
                if info is not chosen:
                    cx, cy = info[1]
                    if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
                        union |= masks_all[info[0]]
            ys, xs = np.nonzero(union)
            hand_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            box_short = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            hand_state = hand_estimator.observe((float(xs.mean()), float(ys.mean())), box_short)
            hand_missing_streak = 0
        else:
            hand_missing_streak += 1
            if hand_missing_streak > tracker_config.max_missing_frames:
                hand_state = None
                hand_estimator.reset()

        row = {"frame": index,
               "hand_conf_max": round(float(confs_all[hand_select].max()), 3) if hand_select.any() else 0.0,
               "hand_state": hand_state is not None,
               "det": int((~hand_select).sum())}
        if hand_state is None:
            row.update({"stage": "hand_state_none", "ranked": 0})
            rows.append(row)
            continue

        masks, class_ids, confidences = masks_all[~hand_select], ids_all[~hand_select], confs_all[~hand_select]
        carved_out = 0
        if hand_conf_map is not None and len(masks) > 0:
            veto = hand_conf_map[None, :, :] > confidences[:, None, None]
            area_before = masks.sum(axis=(1, 2))
            masks = masks & ~veto
            area_after = masks.sum(axis=(1, 2))
            mostly_hand = area_after < 0.5 * np.maximum(area_before, 1)
            if mostly_hand.any():
                carved_out = int(mostly_hand.sum())
                keep = ~mostly_hand
                masks, class_ids, confidences = masks[keep], class_ids[keep], confidences[keep]
        candidates, functional_mask = extract_candidates_from_arrays(
            masks, class_ids, confidences, frame.shape,
            min_confidence=selection_config["candidate"]["min_confidence"],
            min_area_px=selection_config["candidate"]["min_area_px"],
            boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
        )
        extracted = len(candidates)
        candidates, suppressed = suppression_filter.update(candidates, hand_box=hand_box)
        decision = decide_grasp(candidates, functional_mask, hand_state, selection_config)
        held = suppression_filter.stabilize_points(decision.ranked)

        row.update({"stage": "decision", "carved": carved_out, "extracted": extracted,
                    "unstable": suppressed["unstable"], "hand_veto": suppressed["hand_overlap"],
                    "ghost": suppressed.get("ghost", 0), "held": held,
                    "state": decision.state, "ranked": len(decision.ranked),
                    "point": list(decision.ranked[0].grasp_point) if decision.ranked else None})
        rows.append(row)
        if index % 50 == 0:
            print(f"  {index}프레임", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    no_point = [r for r in rows if r["ranked"] == 0]
    ghosts = sum(1 for r in rows if r.get("ghost", 0) > 0)
    print(json.dumps({
        "frames": total,
        "point_missing": len(no_point),
        "missing_by_stage": {s: sum(1 for r in no_point if r["stage"] == s)
                             for s in {r["stage"] for r in no_point}},
        "ghost_frames": ghosts,
        "hand_state_none": sum(1 for r in rows if not r["hand_state"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
