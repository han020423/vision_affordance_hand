#!/usr/bin/env python
"""녹화 영상으로 로봇손 추적기를 오프라인 검증한다.

프레임마다 추적기를 돌려 박스·위치·접근 방향을 그린 영상과 궤적 JSONL,
검출률 요약을 저장한다. 카메라 없이 추적 파라미터를 조정할 때 쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.perception.hand_tracker import HandTracker, load_hand_tracker_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hardware/hand_tracker.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hand_tracking"))
    parser.add_argument("--backend", default=None, help="설정의 backend를 덮어쓴다 (motion 등)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--preview-every", type=int, default=80, help="미리보기 타일 간격(프레임)")
    args = parser.parse_args()

    config = load_hand_tracker_config(args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config)
    if args.backend:
        config.backend = args.backend
    if config.backend == "motion":
        # 배경 모델은 매 프레임 갱신돼야 하므로 움직임 백엔드는 매 프레임 검출한다.
        config.detect_every = 1
    tracker = HandTracker(config, REPOSITORY_ROOT)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    writer = cv2.VideoWriter(str(args.output_dir / f"{stem}_{config.backend}_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trajectory = args.output_dir / f"{stem}_{config.backend}_trajectory.jsonl"
    previews = []
    detected = 0
    total = 0
    with trajectory.open("w", encoding="utf-8") as handle:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames and total >= args.max_frames):
                break
            total += 1
            state = tracker.update(frame)
            overlay = frame.copy()
            if tracker.last_box is not None:
                x1, y1, x2, y2 = tracker.last_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if state is not None:
                detected += 1
                hx, hy = int(state.position[0]), int(state.position[1])
                cv2.drawMarker(overlay, (hx, hy), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
                if state.direction is not None:
                    tip = (int(hx + state.direction[0] * 80), int(hy + state.direction[1] * 80))
                    cv2.arrowedLine(overlay, (hx, hy), tip, (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(overlay, f"frame {total} {'HAND' if state else 'no hand'}", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(overlay)
            handle.write(json.dumps({
                "frame": total,
                "detected": state is not None,
                "position": list(state.position) if state else None,
                "direction": list(state.direction) if (state and state.direction) else None,
                "box": list(tracker.last_box) if tracker.last_box else None,
            }) + "\n")
            if total % args.preview_every == 0:
                previews.append(cv2.resize(overlay, (width // 3, height // 3)))
    writer.release()
    capture.release()
    if previews:
        import numpy as np

        rows = [np.hstack(previews[i:i + 3]) for i in range(0, len(previews) - len(previews) % 3, 3)]
        if rows:
            cv2.imwrite(str(args.output_dir / f"{stem}_{config.backend}_preview.jpg"),
                        np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary = {"frames": total, "detected_frames": detected,
               "detection_rate": round(detected / total, 3) if total else 0.0,
               "backend": config.backend}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
