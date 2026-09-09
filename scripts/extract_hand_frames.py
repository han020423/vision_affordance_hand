#!/usr/bin/env python
"""robot_hand 학습용 영상에서 프레임을 추출한다 (결정적).

data/raw/custom/hand_jetson/*.mp4 를 일정 간격으로 샘플링해
data/raw/custom/hand_jetson_frames/<영상이름>/ 아래 JPEG로 저장하고,
재현을 위해 manifest.jsonl(영상, 프레임 번호, 시각)을 남긴다.

원본 영상은 수정하지 않는다. 같은 인자로 다시 실행하면 같은 결과가 나온다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/raw/custom/hand_jetson")
    parser.add_argument("--output-dir", default="data/raw/custom/hand_jetson_frames")
    parser.add_argument("--every", type=int, default=4, help="N프레임마다 1장 (10fps 영상이면 2.5fps)")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    input_dir = REPOSITORY_ROOT / args.input_dir
    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    records = []
    total = 0
    for video_path in sorted(input_dir.glob("*.mp4")):
        stem = video_path.stem
        frame_dir = output_dir / stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        index = 0
        saved = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % args.every == 0:
                name = f"{stem}_f{index:06d}.jpg"
                cv2.imwrite(
                    str(frame_dir / name), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
                )
                records.append({
                    "image": f"{stem}/{name}",
                    "video": video_path.name,
                    "frame_index": index,
                    "time_s": round(index / fps, 2),
                })
                saved += 1
            index += 1
        capture.release()
        total += saved
        print(f"{stem}: {index}프레임 중 {saved}장 저장")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"합계 {total}장, manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
