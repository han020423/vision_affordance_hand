#!/usr/bin/env python
"""의사라벨 검수 도구 (손 마스크·물체 라벨 공용).

--base 디렉터리의 candidates.jsonl 큐와 overlays/ 오버레이를 한 장씩 보여주고
승인/거부를 기록한다. 의심스러운 프레임을 먼저 보도록 점수가 낮은 순서로
정렬한다. 기본 base는 기존 손 마스크 검수 경로다(하위 호환).

조작:
  y 또는 스페이스  승인 (마스크가 로봇손만 정확히 덮음)
  n               거부 (배경·사람 손이 섞였거나 손이 아님)
  b               한 장 뒤로
  q 또는 ESC      저장하고 종료 (다음 실행 시 이어서)

결과: data/interim/hand_pseudolabels/review.jsonl
승인된 마스크만 학습 데이터로 넘어간다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redo", action="store_true", help="기존 판정을 무시하고 처음부터 다시")
    parser.add_argument("--base", type=Path, default=Path("data/interim/hand_pseudolabels"),
                        help="candidates.jsonl과 overlays/가 있는 디렉터리")
    args = parser.parse_args()

    BASE = args.base if args.base.is_absolute() else REPOSITORY_ROOT / args.base

    candidates = [json.loads(line) for line in (BASE / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [c for c in candidates if c["status"] == "ok"]
    candidates.sort(key=lambda c: (c["box_score"] or 0.0))   # 의심스러운 것부터

    review_path = BASE / "review.jsonl"
    verdicts: dict[str, str] = {}
    if review_path.is_file() and not args.redo:
        for line in review_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                verdicts[row["image"]] = row["verdict"]

    queue = [c for c in candidates if c["image"] not in verdicts]
    print(f"검수 대상 {len(queue)}장 (전체 {len(candidates)}장, 이미 판정 {len(verdicts)}장)")
    if not queue:
        print("모두 판정됐습니다. 다시 하려면 --redo.")
        return 0

    window = "hand mask review (y=승인 n=거부 b=뒤로 q=종료)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    index = 0
    history: list[str] = []
    while 0 <= index < len(queue):
        item = queue[index]
        image_path = BASE / "overlays" / item["image"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            index += 1
            continue
        done = len(verdicts)
        header = (f"[{done + 1}/{len(candidates)}] score={item['box_score']:.2f} "
                  f"mask={item['mask_px']}px  {Path(item['image']).name}")
        cv2.putText(frame, header, (10, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord(" ")):
            verdicts[item["image"]] = "approved"
            history.append(item["image"])
            index += 1
        elif key == ord("n"):
            verdicts[item["image"]] = "rejected"
            history.append(item["image"])
            index += 1
        elif key == ord("b") and history:
            previous = history.pop()
            verdicts.pop(previous, None)
            index -= 1
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    with review_path.open("w", encoding="utf-8") as handle:
        for image, verdict in sorted(verdicts.items()):
            handle.write(json.dumps({"image": image, "verdict": verdict}, ensure_ascii=False) + "\n")
    approved = sum(1 for v in verdicts.values() if v == "approved")
    rejected = sum(1 for v in verdicts.values() if v == "rejected")
    print(f"저장: {review_path}")
    print(f"승인 {approved} / 거부 {rejected} / 남음 {len(candidates) - len(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
