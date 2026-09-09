#!/usr/bin/env python
"""카메라 원본 영상을 녹화한다 (오버레이 없음 — 학습 데이터 촬영용).

robot_hand 클래스 학습 데이터 촬영을 위해 만들었다. run_realtime_rfdetr.py의
--record는 검출 표시가 그려진 overlay를 저장하므로 학습 원본으로 쓸 수 없다.

사용 예 (젯슨):
    .venv/bin/python scripts/record_camera.py --duration 30
    .venv/bin/python scripts/record_camera.py --duration 30 --note hand_still

저장 위치: data/raw/custom/hand_jetson/hand_<날짜시각>[_note].mp4
조작: q 또는 ESC = 조기 종료 (그때까지 저장분은 유지)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=30.0, help="녹화 시간(초)")
    parser.add_argument("--note", default="", help="파일 이름에 붙일 짧은 메모 (예: hand_still)")
    parser.add_argument("--output-dir", default="data/raw/custom/hand_jetson")
    parser.add_argument("--no-window", action="store_true", help="미리보기 창 없이 녹화")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"카메라 {args.camera_index}번을 열 수 없습니다.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 30.0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.note}" if args.note else ""
    video_path = output_dir / f"hand_{stamp}{suffix}.mp4"

    writer = None
    frames = 0
    started = time.monotonic()
    print(f"녹화 시작: {video_path.name} ({args.duration:.0f}초, {fps:.0f}fps 목표)")
    try:
        while time.monotonic() - started < args.duration:
            ok, frame = capture.read()
            if not ok:
                continue
            if writer is None:
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (frame.shape[1], frame.shape[0]),
                )
            writer.write(frame)          # 원본 그대로 저장 (표시 없음)
            frames += 1

            if not args.no_window:
                preview = frame.copy()   # 미리보기에만 정보 표시, 저장본은 원본
                remaining = args.duration - (time.monotonic() - started)
                cv2.putText(preview, f"REC {remaining:4.0f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("record (q=stop)", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    seconds = time.monotonic() - started
    print(f"저장: {video_path}  ({frames}프레임, {seconds:.1f}초, 실측 {frames / max(seconds, 0.1):.1f}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
