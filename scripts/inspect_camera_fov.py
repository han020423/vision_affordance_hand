#!/usr/bin/env python
"""Windows에서 연결된 카메라 번호와 실제 화각 스냅샷을 점검한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def inspect_camera(index: int, output_dir: Path, warmup_frames: int) -> dict[str, object]:
    """카메라 하나를 열어 해상도를 기록하고 마지막 준비 프레임을 저장한다."""

    # OpenCV 배포본과 카메라 드라이버에 따라 지원 backend가 다르므로 순서대로 시도한다.
    camera = cv2.VideoCapture()
    backend_name = None
    for name, backend in (("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("자동", cv2.CAP_ANY)):
        if camera.open(index, backend):
            backend_name = name
            break
        camera.release()
        camera = cv2.VideoCapture()
    result: dict[str, object] = {"camera_index": index, "opened": camera.isOpened()}
    if not camera.isOpened():
        camera.release()
        return result

    ok = False
    frame = None
    for _ in range(warmup_frames):
        ok, frame = camera.read()
        if not ok:
            break

    result.update(
        {
            "read_ok": bool(ok),
            "backend": backend_name,
            "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(camera.get(cv2.CAP_PROP_FPS)),
        }
    )
    if ok and frame is not None:
        snapshot = output_dir / f"camera_{index}.jpg"
        if not cv2.imwrite(str(snapshot), frame):
            raise OSError(f"카메라 스냅샷을 저장하지 못했습니다: {snapshot}")
        result["snapshot"] = str(snapshot.resolve())
    camera.release()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="카메라 번호별 화각 스냅샷을 저장합니다.")
    parser.add_argument("--max-index", type=int, default=4, help="검사할 마지막 카메라 번호")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/camera_diagnostics"),
        help="진단 결과를 저장할 폴더",
    )
    parser.add_argument("--warmup-frames", type=int, default=15)
    args = parser.parse_args()
    if args.max_index < 0 or args.warmup_frames < 1:
        raise ValueError("카메라 번호는 0 이상, 준비 프레임 수는 1 이상이어야 합니다")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        inspect_camera(index, args.output_dir, args.warmup_frames)
        for index in range(args.max_index + 1)
    ]
    report_path = args.output_dir / "camera_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"카메라 진단 보고서: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
