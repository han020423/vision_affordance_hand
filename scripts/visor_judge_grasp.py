#!/usr/bin/env python
"""VISOR 접촉 장면을 사람이 파지 유형으로 판정하는 GUI 도구.

렌더링된 오버레이(노랑=물체, 빨강 윤곽=손)를 보면서 키 하나로 판정한다.
결과는 JSONL로 저장되며 중단 후 재실행하면 이어서 진행된다.

판정 키:
  1: 손잡이 파지        (handle)
  2: 몸통 파지          (body)
  3: 테두리·꼭지 집기   (rim_pinch)
  4: 기능부 접촉        (functional_touch)
  5: 파지 아님·판단불가 (not_grasp)
  B: 이전으로  /  Q: 종료(저장됨)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

LABELS = {
    ord("1"): "handle",
    ord("2"): "body",
    ord("3"): "rim_pinch",
    ord("4"): "functional_touch",
    ord("5"): "not_grasp",
}
LABEL_KOREAN = {
    "handle": "손잡이 파지",
    "body": "몸통 파지",
    "rim_pinch": "테두리·꼭지 집기",
    "functional_touch": "기능부 접촉",
    "not_grasp": "파지 아님",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="judgment_manifest.jsonl과 이미지가 있는 폴더")
    args = parser.parse_args()

    manifest_path = args.workspace / "judgment_manifest.jsonl"
    result_path = args.workspace / "judgments.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    done: dict[int, dict] = {}
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[int(row["index"])] = row

    def save_all() -> None:
        with result_path.open("w", encoding="utf-8") as handle:
            for key in sorted(done):
                handle.write(json.dumps(done[key], ensure_ascii=False) + "\n")

    pending = [item for item in items if item["index"] not in done]
    print(f"전체 {len(items)}장 중 완료 {len(done)}장, 남은 {len(pending)}장")
    print("키: 1=손잡이 2=몸통 3=테두리집기 4=기능부 5=파지아님 | B=이전 Q=종료")

    position = 0
    window = "VISOR 파지 판정"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while position < len(pending):
        item = pending[position]
        image = cv2.imread(str(args.workspace / item["file"]))
        if image is None:
            position += 1
            continue
        header = f"[{len(done)}/{len(items)}] {item['category']} ({item['object_name']})"
        display = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(display, header, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("b"), ord("B")) and position > 0:
            position -= 1
            previous = pending[position]
            done.pop(previous["index"], None)
            continue
        if key in LABELS:
            label = LABELS[key]
            done[item["index"]] = {**item, "judgment": label}
            save_all()
            position += 1
    cv2.destroyAllWindows()
    save_all()
    print(f"저장 완료: {result_path} (판정 {len(done)}/{len(items)}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
