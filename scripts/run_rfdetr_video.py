#!/usr/bin/env python
"""녹화 영상으로 RF-DETR-Seg 체크포인트를 오프라인 검증한다 (YOLO 비교용).

run_realtime_rfdetr.py의 렌더링을 재사용하되 카메라 대신 영상 파일을 읽고,
프레임별 클래스 검출 여부·추론 시간·미리보기 타일·overlay 영상을 저장한다.
hjh_rfdetr conda 환경에서 실행한다.
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

CLASSES = {0: ("handle", (0, 200, 0)), 1: ("body", (255, 120, 0)), 2: ("functional", (0, 0, 255))}


def render_overlay(frame, detections, class_id_base: int, alpha: float):
    """RF-DETR 마스크를 uint8 프레임에 채색한다 (OpenCV 5 호환: 텍스트는 uint8에만 그린다)."""

    counts = {name: 0 for name, _ in CLASSES.values()}
    if detections is None or detections.mask is None or len(detections) == 0:
        return frame.copy(), counts
    height, width = frame.shape[:2]
    output = frame.astype(np.float32)
    labels = []
    for mask, raw_id, confidence in zip(np.asarray(detections.mask),
                                        np.asarray(detections.class_id, dtype=int),
                                        np.asarray(detections.confidence)):
        class_id = int(raw_id) - class_id_base
        if class_id not in CLASSES:
            continue
        name, color = CLASSES[class_id]
        counts[name] += 1
        mask = mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        region = mask.astype(bool)
        output[region] = output[region] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, _, _ = cv2.boundingRect(contours[0])
            labels.append((f"{name} {confidence:.2f}", (x, max(15, y - 5)), color, contours))
    rendered = np.clip(output, 0, 255).astype(np.uint8)
    for text, origin, color, contours in labels:
        cv2.drawContours(rendered, contours, -1, color, 2)
        cv2.putText(rendered, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return rendered, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("outputs/training/custom_finetune_k_rfdetr_seg_seed42/checkpoint_best_ema.pth"))
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rfdetr_video_check"))
    parser.add_argument("--preview-frames", default="60,150,200,260,330,400")
    args = parser.parse_args()

    import rfdetr

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
    writer = cv2.VideoWriter(str(args.output_dir / f"{stem}_rfdetr_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    preview_ids = {int(v) for v in args.preview_frames.split(",") if v.strip()}
    previews: list[np.ndarray] = []
    per_frame = []
    times = []
    class_id_base = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        t0 = time.perf_counter()
        detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
        times.append((time.perf_counter() - t0) * 1000.0)
        if detections is not None and len(detections) > 0:
            observed_max = int(np.asarray(detections.class_id, dtype=int).max())
            class_id_base = 1 if observed_max >= len(CLASSES) else 0
        overlay, counts = render_overlay(frame, detections, class_id_base, 0.45)
        cv2.putText(overlay, f"RF-DETR frame {frame_index} infer {times[-1]:.0f}ms", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(overlay)
        per_frame.append({"frame": frame_index, **counts})
        if frame_index in preview_ids:
            tile = cv2.resize(overlay, (960, 540))
            cv2.putText(tile, f"frame {frame_index}", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            previews.append(tile)
    writer.release()
    capture.release()

    if previews:
        rows = [np.hstack(previews[i:i + 2]) for i in range(0, len(previews) - len(previews) % 2, 2)]
        if rows:
            cv2.imwrite(str(args.output_dir / f"{stem}_rfdetr_preview.jpg"), np.vstack(rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.output_dir / f"{stem}_rfdetr_per_frame.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_frame) + "\n", encoding="utf-8")
    total = len(per_frame)
    summary = {
        "frames": total,
        "frames_with_body": sum(1 for r in per_frame if r["body"] > 0),
        "frames_with_handle": sum(1 for r in per_frame if r["handle"] > 0),
        "frames_with_functional": sum(1 for r in per_frame if r["functional"] > 0),
        "mean_infer_ms": round(float(np.mean(times)), 1) if times else None,
        "conf": args.conf,
        "variant": args.variant,
    }
    (args.output_dir / f"{stem}_rfdetr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
