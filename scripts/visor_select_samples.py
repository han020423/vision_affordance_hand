#!/usr/bin/env python
"""VISOR 손-물체 접촉 이벤트에서 파지 판정용 표본을 층화 추출한다.

교수님 피드백(인간 파지 행동 근거) 대응 분석의 1단계 도구다.
- 1층(관심 물체): 후보 점수 prior 산출용
- 2층(보강 물체): 집기(PRECISION) 계열 행동 근거·논문 표 전용
- 물체 이름은 개방 어휘이므로 "이름의 마지막 단어" 일치로 분류한다
  (예: 'oil bottle'→bottle, 'cupboard'는 cup에 불일치, 'potato'는 pot에 불일치)

출력:
- samples.jsonl: 판정 대상 이벤트 (프레임·손/물체 마스크 폴리곤 포함)
- videos_needed.txt: 이미지 zip을 받아야 하는 비디오 목록
- selection_summary.json: 물체별 추출 수량
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# 1층: 프로젝트 관심 물체 (prior 주입용)
TIER1_KEYWORDS = [
    "mug", "cup", "kettle", "pan", "pot", "knife", "scissors", "spoon",
    "ladle", "spatula", "bottle", "glass", "bowl",
]
# 2층: 집기 계열 보강 (논문 근거 전용)
TIER2_KEYWORDS = ["plate", "lid", "fork", "jar", "container", "colander", "board"]

NOT_CONTACT = {None, "", "hand-not-in-contact", "none-of-the-above", "inconclusive",
               "glove-not-in-contact"}


def category_of(name: str) -> tuple[str, int] | None:
    """물체 이름의 마지막 단어로 (범주, 층)을 정한다. 해당 없으면 None."""

    last_word = name.strip().lower().split()[-1] if name.strip() else ""
    if last_word in TIER1_KEYWORDS:
        return last_word, 1
    if last_word in TIER2_KEYWORDS:
        return last_word, 2
    return None


def collect_events(annotation_root: Path) -> list[dict]:
    """접촉이 확정된 손-물체 이벤트를 전부 수집한다."""

    events: list[dict] = []
    for split in ("train", "val"):
        for path in sorted((annotation_root / split).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            video = path.stem
            for frame in data["video_annotations"]:
                by_id = {a["id"]: a for a in frame["annotations"]}
                image = frame["image"]
                for ann in frame["annotations"]:
                    hand_name = str(ann.get("name", ""))
                    if "hand" not in hand_name and "glove" not in hand_name:
                        continue
                    contact = ann.get("in_contact_object")
                    if contact in NOT_CONTACT:
                        continue
                    target = by_id.get(contact)
                    if target is None:
                        continue
                    matched = category_of(str(target.get("name", "")))
                    if matched is None:
                        continue
                    category, tier = matched
                    events.append(
                        {
                            "split": split,
                            "video": video,
                            "image_name": image["name"],
                            "image_path": image["image_path"],
                            "category": category,
                            "tier": tier,
                            "object_name": str(target.get("name", "")),
                            "hand_name": hand_name,
                            "object_segments": target.get("segments", []),
                            "hand_segments": ann.get("segments", []),
                        }
                    )
    return events


def stratified_sample(events: list[dict], quota1: int, quota2: int, seed: int) -> list[dict]:
    """범주별 정원을 비디오 라운드로빈으로 채워 한 참가자 편중을 줄인다."""

    rng = random.Random(seed)
    selected: list[dict] = []
    by_category: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_category[event["category"]].append(event)
    for category, items in sorted(by_category.items()):
        quota = quota1 if items[0]["tier"] == 1 else quota2
        by_video: dict[str, list[dict]] = defaultdict(list)
        for event in items:
            by_video[event["video"]].append(event)
        for video_events in by_video.values():
            rng.shuffle(video_events)
        videos = sorted(by_video)
        rng.shuffle(videos)
        picked: list[dict] = []
        round_index = 0
        while len(picked) < min(quota, len(items)):
            advanced = False
            for video in videos:
                queue = by_video[video]
                if round_index < len(queue):
                    picked.append(queue[round_index])
                    advanced = True
                    if len(picked) >= min(quota, len(items)):
                        break
            if not advanced:
                break
            round_index += 1
        selected.extend(picked)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quota-tier1", type=int, default=50)
    parser.add_argument("--quota-tier2", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    events = collect_events(args.annotation_root)
    selected = stratified_sample(events, args.quota_tier1, args.quota_tier2, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for event in selected:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    videos = sorted({(event["split"], event["video"]) for event in selected})
    (args.output_dir / "videos_needed.txt").write_text(
        "\n".join(f"{split}/{video}" for split, video in videos) + "\n", encoding="utf-8"
    )
    summary = {
        "total_events_found": len(events),
        "selected": len(selected),
        "videos_needed": len(videos),
        "per_category": dict(sorted(Counter(e["category"] for e in selected).items())),
        "per_tier": dict(Counter(e["tier"] for e in selected)),
        "seed": args.seed,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
