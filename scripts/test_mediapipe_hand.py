#!/usr/bin/env python
"""MediaPipe Hands가 로봇 의수(검은색, 사람 손 형태)를 인식하는지 실측한다.

웹캠 프레임마다 손 랜드마크를 검출해 검출률, 신뢰도, 손목 좌표 떨림,
프레임당 처리 시간을 기록한다. 결과는 outputs/mediapipe_hand_test 에
영상·요약 JSON으로 저장한다. 조작: q/ESC 종료, s 스냅숏.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--duration", type=float, default=0.0, help="자동 종료 초(0=q로만 종료)")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mediapipe_hand_test"))
    args = parser.parse_args()

    import mediapipe as mp

    hands_module = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera_index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"카메라 {args.camera_index}번을 열 수 없습니다.")

    frames = 0
    detected_frames = 0
    scores: list[float] = []
    latencies_ms: list[float] = []
    wrist_track: list[tuple[float, float]] = []   # 연속 검출 구간의 손목 픽셀 좌표
    jitter_samples: list[float] = []              # 연속 프레임 간 손목 이동 거리(px)
    previous_wrist: tuple[float, float] | None = None
    writer = None
    stop_reason = "user_quit"
    start = time.perf_counter()

    with hands_module.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    continue
                frames += 1
                t0 = time.perf_counter()
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                latency = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency)

                height, width = frame.shape[:2]
                if result.multi_hand_landmarks:
                    detected_frames += 1
                    for landmarks, handedness in zip(
                        result.multi_hand_landmarks, result.multi_handedness or []
                    ):
                        drawing.draw_landmarks(
                            frame,
                            landmarks,
                            hands_module.HAND_CONNECTIONS,
                            styles.get_default_hand_landmarks_style(),
                            styles.get_default_hand_connections_style(),
                        )
                        score = float(handedness.classification[0].score)
                        label = handedness.classification[0].label
                        scores.append(score)
                        wrist = landmarks.landmark[0]
                        wx, wy = wrist.x * width, wrist.y * height
                        cv2.putText(
                            frame, f"{label} {score:.2f}", (int(wx) + 8, int(wy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        )
                    # 첫 번째 손의 손목으로 떨림을 잰다(연속 검출일 때만).
                    first = result.multi_hand_landmarks[0].landmark[0]
                    current = (first.x * width, first.y * height)
                    if previous_wrist is not None:
                        jitter_samples.append(float(np.hypot(
                            current[0] - previous_wrist[0], current[1] - previous_wrist[1]
                        )))
                    previous_wrist = current
                    wrist_track.append(current)
                else:
                    previous_wrist = None

                rate = detected_frames / frames * 100.0
                cv2.putText(
                    frame,
                    f"detect {detected_frames}/{frames} ({rate:.0f}%)  {latency:.0f}ms  conf>={args.min_detection_confidence}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

                if args.record:
                    if writer is None:
                        video_path = output_dir / f"mediapipe_hand_{stamp}.mp4"
                        writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height)
                        )
                    writer.write(frame)

                cv2.imshow("MediaPipe hand test (prosthetic)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    snap = output_dir / f"mediapipe_snapshot_{stamp}_{frames:05d}.png"
                    cv2.imwrite(str(snap), frame)
                    print(f"스냅숏 저장: {snap}")
                if args.duration > 0 and (time.perf_counter() - start) >= args.duration:
                    stop_reason = "duration_elapsed"
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    summary = {
        "camera_index": args.camera_index,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "frames": frames,
        "detected_frames": detected_frames,
        "detection_rate": round(detected_frames / frames, 4) if frames else None,
        "mean_handedness_score": round(float(np.mean(scores)), 4) if scores else None,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 1) if latencies_ms else None,
        "wrist_jitter_px_mean": round(float(np.mean(jitter_samples)), 2) if jitter_samples else None,
        "wrist_jitter_px_p90": round(float(np.percentile(jitter_samples, 90)), 2) if jitter_samples else None,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "stop_reason": stop_reason,
        "note": "검출률은 테스트 중 손이 화면에 있던 비율에 따라 달라지므로 영상과 함께 해석한다.",
    }
    summary_path = output_dir / f"mediapipe_hand_{stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
