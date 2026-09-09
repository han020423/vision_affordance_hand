#!/usr/bin/env python
"""M(RF-DETR-Seg) + 분리형 손 검출기 결합 파이프라인의 오프라인 검증 프로그램.

실시간 프로그램(run_realtime_rfdetr.py --hand-source detector)과 같은 흐름을
카메라 대신 녹화 영상으로 재현한다: M이 물체 affordance를 분할하고, 별도
손 검출기(YOLO11n, backend=custom)가 손 위치·접근 방향을 추적하며, 억제 필터와
점수식이 파지 후보·자세(GRASP/ALIGN/NO_TARGET, 집기/감아쥐기/움켜쥐기)를 결정한다.

하드웨어 없이 전체 통합을 검증하는 것이 목적이므로 프레임별 결정 로그(jsonl)와
overlay 영상, 요약 JSON을 남긴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

# 실시간 프로그램과 동일 로직을 쓰기 위해 헬퍼를 재사용한다(중복 구현 금지).
from run_realtime_rfdetr import (  # noqa: E402
    ROBOT_HAND_CLASS,
    detections_to_arrays,
    render_overlay,
    resolve_repo_path,
)
from src.grasp_selection import (  # noqa: E402
    HandSuppressionFilter,
    decide_grasp,
    extract_candidates_from_arrays,
    load_selection_config,
)
from src.perception.hand_tracker import HandTracker, load_hand_tracker_config  # noqa: E402
from src.perception.realtime import draw_selection, load_realtime_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path("outputs/training/custom_finetune_m_rfdetr_customv3_seed42/checkpoint_best_ema.pth"))
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--config", default="configs/hardware/realtime_camera.yaml")
    parser.add_argument("--hand-tracker-config", default="configs/hardware/hand_tracker_detector.yaml")
    parser.add_argument("--selection-config", default="configs/grasp_selection.yaml")
    parser.add_argument("--human-prior-config", default="configs/human_grasp_prior.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_pipeline"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    import rfdetr

    config = load_realtime_config(resolve_repo_path(args.config))
    selection_config = load_selection_config(
        resolve_repo_path(args.selection_config), resolve_repo_path(args.human_prior_config))
    tracker_config = load_hand_tracker_config(resolve_repo_path(args.hand_tracker_config))
    hand_tracker = HandTracker(tracker_config, REPOSITORY_ROOT)
    suppression_filter = HandSuppressionFilter.from_config(selection_config)

    model_path = args.model if args.model.is_absolute() else REPOSITORY_ROOT / args.model
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(model_path))

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    writer = cv2.VideoWriter(str(args.output_dir / f"{stem}_pipeline.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    log_path = args.output_dir / f"{stem}_pipeline.jsonl"
    log_file = log_path.open("w", encoding="utf-8")

    class_id_base = 0
    frame_index = 0
    infer_times: list[float] = []
    state_counts: dict[str, int] = {}
    pose_counts: dict[str, int] = {}
    hand_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if args.max_frames and frame_index > args.max_frames:
            break

        t0 = time.perf_counter()
        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   threshold=config.conf_threshold)
        infer_times.append((time.perf_counter() - t0) * 1000.0)
        if detections is not None and len(detections) > 0:
            class_id_base = 0 if int(np.asarray(detections.class_id, dtype=int).min()) == 0 else 1

        overlay, _ = render_overlay(frame, detections, class_id_base, config.overlay_alpha)

        hand_state = hand_tracker.update(frame)
        if hand_state is not None:
            hand_frames += 1
        if hand_tracker.last_box is not None and hand_state is not None:
            bx1, by1, bx2, by2 = hand_tracker.last_box
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.putText(overlay, "robot hand", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "HAND NOT DETECTED", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        decision = None
        if hand_state is not None:
            masks, class_ids, confidences = detections_to_arrays(detections, class_id_base)
            if len(masks) > 0:
                keep = class_ids != ROBOT_HAND_CLASS
                masks, class_ids, confidences = masks[keep], class_ids[keep], confidences[keep]
            candidates, functional_mask = extract_candidates_from_arrays(
                masks, class_ids, confidences, frame.shape,
                min_confidence=selection_config["candidate"]["min_confidence"],
                min_area_px=selection_config["candidate"]["min_area_px"],
                boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
            )
            hand_box = hand_tracker.last_box
            candidates, _ = suppression_filter.update(candidates, hand_box=hand_box)
            decision = decide_grasp(candidates, functional_mask, hand_state, selection_config)
            draw_selection(overlay, decision, hand_state)
            state_counts[decision.state] = state_counts.get(decision.state, 0) + 1
            if decision.state == "GRASP" and decision.pose:
                pose_counts[decision.pose] = pose_counts.get(decision.pose, 0) + 1
        else:
            state_counts["NO_HAND"] = state_counts.get("NO_HAND", 0) + 1

        cv2.putText(overlay, f"frame {frame_index} infer {infer_times[-1]:.0f}ms",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(overlay)
        log_file.write(json.dumps({
            "frame": frame_index,
            "hand": hand_state is not None,
            "state": decision.state if decision else None,
            "pose": decision.pose if decision else None,
            "score": (round(float(decision.candidate.scores["total"]), 3)
                      if decision and decision.candidate else None),
            "target": decision.candidate.class_name if decision and decision.candidate else None,
        }, ensure_ascii=False) + "\n")

    capture.release()
    writer.release()
    log_file.close()

    summary = {
        "video": str(args.video), "frames": frame_index,
        "hand_detected_frames": hand_frames,
        "hand_rate": round(hand_frames / max(frame_index, 1), 3),
        "state_counts": state_counts, "pose_counts_during_grasp": pose_counts,
        "mean_infer_ms": round(float(np.mean(infer_times)), 1) if infer_times else None,
        "model": str(args.model), "conf_threshold": config.conf_threshold,
        "hand_backend": tracker_config.backend,
    }
    (args.output_dir / f"{stem}_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
