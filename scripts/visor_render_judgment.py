#!/usr/bin/env python
"""VISOR 표본 이벤트를 사람 판정용 오버레이 이미지로 렌더링한다.

samples.jsonl의 각 이벤트에 대해 프레임을 zip에서 읽어
물체 마스크(노랑 채움)와 손 마스크(빨강 윤곽)를 겹치고, 물체 주변을
잘라 판정하기 쉬운 크기로 저장한다. 원본 zip은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def polygons_from_segments(segments) -> list[np.ndarray]:
    """VISOR segments 필드를 폴리곤 배열 목록으로 해석한다."""

    polygons: list[np.ndarray] = []
    for segment in segments or []:
        points = np.asarray(segment, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            polygons.append(points.astype(np.int32))
    return polygons


def read_frame(zip_root: Path, video: str, image_name: str, cache: dict) -> np.ndarray | None:
    """비디오 zip에서 프레임 한 장을 읽는다.

    파일 핸들 고갈을 막기 위해 zip은 한 번에 하나만 열어 둔다. 손상되었거나
    열 수 없는 zip은 실패 목록에 기록하고 해당 이벤트를 건너뛴다.
    """

    zip_path = zip_root / f"{video}.zip"
    if not zip_path.is_file():
        cache.setdefault("failed", set()).add(video)
        return None
    if video in cache.get("failed", set()):
        return None
    current = cache.get("current")
    if current is None or current[0] != video:
        if current is not None:
            current[1].close()
            cache["current"] = None
        try:
            handle = zipfile.ZipFile(zip_path)
            names = {Path(name).name: name for name in handle.namelist()}
        except (zipfile.BadZipFile, OSError) as exc:
            cache.setdefault("failed", set()).add(video)
            print(f"[손상 zip 건너뜀] {video}: {type(exc).__name__}")
            return None
        cache["current"] = (video, handle, names)
    _, handle, names = cache["current"]
    member = names.get(image_name)
    if member is None:
        return None
    try:
        data = np.frombuffer(handle.read(member), dtype=np.uint8)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[프레임 읽기 실패] {video}/{image_name}: {type(exc).__name__}")
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def render_event(frame: np.ndarray, event: dict, alpha: float = 0.35) -> np.ndarray:
    """물체(노랑 채움)·손(빨강 윤곽)을 표시하고 물체 주변을 잘라낸다."""

    overlay = frame.copy()
    object_polys = polygons_from_segments(event["object_segments"])
    hand_polys = polygons_from_segments(event["hand_segments"])
    layer = overlay.copy()
    for poly in object_polys:
        cv2.fillPoly(layer, [poly], (0, 220, 255))
    overlay = cv2.addWeighted(layer, alpha, overlay, 1 - alpha, 0)
    for poly in object_polys:
        cv2.polylines(overlay, [poly], True, (0, 220, 255), 2)
    for poly in hand_polys:
        cv2.polylines(overlay, [poly], True, (0, 0, 255), 3)

    # 물체와 손을 모두 포함하는 영역을 여유를 두고 잘라 판정을 쉽게 한다.
    all_points = np.vstack([p for p in object_polys + hand_polys]) if (object_polys or hand_polys) else None
    if all_points is not None:
        x, y, w, h = cv2.boundingRect(all_points)
        margin = int(0.25 * max(w, h))
        x0 = max(0, x - margin); y0 = max(0, y - margin)
        x1 = min(frame.shape[1], x + w + margin); y1 = min(frame.shape[0], y + h + margin)
        overlay = overlay[y0:y1, x0:x1]
    height, width = overlay.shape[:2]
    scale = 720 / max(height, width)
    if scale < 1:
        overlay = cv2.resize(overlay, (int(width * scale), int(height * scale)))
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--zip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    manifest = []
    missing = 0
    with args.samples.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    # zip을 한 번에 하나만 열도록 비디오 기준으로 정렬해 처리한다.
    order = sorted(range(len(events)), key=lambda i: events[i]["video"])
    for index in order:
        event = events[index]
        frame = read_frame(args.zip_root, event["video"], event["image_name"], cache)
        if frame is None:
            missing += 1
            continue
        rendered = render_event(frame, event)
        filename = f"{index:04d}_{event['category']}.jpg"
        cv2.imwrite(str(args.output_dir / filename), rendered, [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest.append(
            {
                "index": index,
                "file": filename,
                "category": event["category"],
                "tier": event["tier"],
                "video": event["video"],
                "image_name": event["image_name"],
                "object_name": event["object_name"],
                "hand_name": event["hand_name"],
            }
        )
    manifest.sort(key=lambda row: row["index"])
    with (args.output_dir / "judgment_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    failed_videos = sorted(cache.get("failed", set()))
    if failed_videos:
        (args.output_dir / "failed_videos.txt").write_text(
            "\n".join(failed_videos) + "\n", encoding="utf-8"
        )
    print(f"렌더링 {len(manifest)}장 완료, 프레임 누락 {missing}건, 실패 zip {len(failed_videos)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
