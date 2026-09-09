#!/usr/bin/env python
"""VISOR 파지 판정 표본의 기하 특징을 계산한다 (파지 자세 근거 분석).

목적: "폭 40px" 같은 임의 임계값 대신, 사람이 실제로 한 파지(868장면 판정)에서
자세 결정의 기하적 근거를 추출한다. Feix의 방법론(물체 치수 ↔ 파지 유형 상관)을
따르되, VISOR의 손·물체 분할 마스크 덕분에 손 크기로 정규화한 치수를 쓸 수 있다.
정규화 비율은 카메라 거리와 무관하므로 실행 시에도 같은 특징을 쓸 수 있다.

입력:
- outputs/visor_judgment/judgments.jsonl  (868건 사람 판정: video, image_name,
  object_name, hand_name, judgment)
- data/raw/visor/annotations/{train,val}/<video>.json  (원본 폴리곤 주석)

표본별 계산 특징 (해상도 1920x1080 기준 픽셀):
- obj_minor / obj_major / obj_aspect : 물체 전체 최소면적사각형의 짧은/긴 변, 비율
- obj_thick        : 물체 마스크 distance transform 최댓값×2 (가장 굵은 부분)
- contact_thick    : 손 주변 접촉 대역에서의 물체 국소 두께(DT 최댓값×2)
                     — "실제로 잡은 부위의 굵기"에 해당 (Feix의 grasped dimension)
- hand_palm        : 손 마스크 DT 최댓값×2 (손바닥 폭 근사; 팔뚝이 섞여도 안정적)
- hand_minor       : 손 최소면적사각형 짧은 변
- r_contact = contact_thick / hand_palm   ← 주 특징 (거리 무관)
- r_minor   = obj_minor / hand_palm
- contact_px       : 접촉 대역 픽셀 수(신뢰도 지표)

출력:
- outputs/visor_geometry/features.csv
- outputs/visor_geometry/summary.json  (판정 범주별 분포 통계 + 경계 탐색)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = REPOSITORY_ROOT / "outputs" / "visor_judgment" / "judgments.jsonl"
ANNOTATION_ROOT = REPOSITORY_ROOT / "data" / "raw" / "visor" / "annotations"
OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "visor_geometry"

WIDTH, HEIGHT = 1920, 1080


def rasterize(segments) -> np.ndarray:
    """폴리곤 목록을 bool 마스크로 그린다."""

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for poly in segments or []:
        if not poly or len(poly) < 3:
            continue
        points = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, WIDTH - 1)
        points[:, 1] = np.clip(points[:, 1], 0, HEIGHT - 1)
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def min_rect_sides(mask: np.ndarray) -> tuple[float, float]:
    """최소면적사각형의 (짧은 변, 긴 변)."""

    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0, 0.0
    points = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    return float(min(w, h)), float(max(w, h))


def thickness(mask: np.ndarray) -> np.ndarray:
    """distance transform (내접 반지름 지도)."""

    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def find_event(frames_by_name: dict, record: dict):
    """판정 레코드에 해당하는 (손 주석, 물체 주석)을 찾는다."""

    frame = frames_by_name.get(record["image_name"])
    if frame is None:
        return None, None, "frame_missing"
    by_id = {a["id"]: a for a in frame["annotations"]}
    candidates = []
    for ann in frame["annotations"]:
        if str(ann.get("name", "")) != record["hand_name"]:
            continue
        target = by_id.get(ann.get("in_contact_object"))
        if target is None:
            continue
        if str(target.get("name", "")) == record["object_name"]:
            candidates.append((ann, target))
    if not candidates:
        return None, None, "event_missing"
    return candidates[0][0], candidates[0][1], ""


def main() -> int:
    records = [json.loads(line) for line in JUDGMENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"판정 레코드 {len(records)}건")
    counts = defaultdict(int)
    for r in records:
        counts[r["judgment"]] += 1
    print("판정 분포:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))

    # 비디오별로 묶어 주석 JSON을 한 번씩만 연다.
    by_video = defaultdict(list)
    for r in records:
        by_video[r["video"]].append(r)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = defaultdict(int)
    for vi, (video, recs) in enumerate(sorted(by_video.items()), 1):
        path = None
        for split in ("train", "val"):
            candidate = ANNOTATION_ROOT / split / f"{video}.json"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            skipped["annotation_missing"] += len(recs)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        frames_by_name = {f["image"]["name"]: f for f in data["video_annotations"]}

        for record in recs:
            hand_ann, obj_ann, error = find_event(frames_by_name, record)
            if error:
                skipped[error] += 1
                continue
            hand = rasterize(hand_ann.get("segments"))
            obj = rasterize(obj_ann.get("segments"))
            if hand.sum() < 100 or obj.sum() < 100:
                skipped["mask_too_small"] += 1
                continue

            obj_minor, obj_major = min_rect_sides(obj)
            hand_minor, hand_major = min_rect_sides(hand)
            obj_dt = thickness(obj)
            hand_dt = thickness(hand)
            obj_thick = 2.0 * float(obj_dt.max())
            hand_palm = 2.0 * float(hand_dt.max())

            # 접촉 대역: 손을 넓혀 물체와 겹치는 부분. 안 겹치면 반경을 키운다.
            contact = None
            for radius in (15, 30, 60):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                dilated = cv2.dilate(hand.astype(np.uint8), kernel).astype(bool)
                band = dilated & obj
                if band.sum() >= 30:
                    contact = band
                    break
            if contact is None:
                skipped["no_contact_band"] += 1
                continue
            contact_thick = 2.0 * float(obj_dt[contact].max())

            rows.append({
                "index": record["index"],
                "judgment": record["judgment"],
                "category": record["category"],
                "tier": record["tier"],
                "video": video,
                "object_name": record["object_name"],
                "obj_minor": round(obj_minor, 1),
                "obj_major": round(obj_major, 1),
                "obj_aspect": round(obj_major / obj_minor, 2) if obj_minor > 0 else 0.0,
                "obj_thick": round(obj_thick, 1),
                "contact_thick": round(contact_thick, 1),
                "hand_palm": round(hand_palm, 1),
                "hand_minor": round(hand_minor, 1),
                "contact_px": int(contact.sum()),
                "r_contact": round(contact_thick / hand_palm, 3) if hand_palm > 0 else 0.0,
                "r_minor": round(obj_minor / hand_palm, 3) if hand_palm > 0 else 0.0,
            })
        if vi % 20 == 0:
            print(f"  {vi}/{len(by_video)} 비디오 처리, 표본 {len(rows)}건")

    print(f"\n특징 계산 완료: {len(rows)}건, 제외 {dict(skipped)}")

    csv_path = OUTPUT_DIR / "features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {csv_path}")

    # ---- 범주별 분포와 경계 탐색 ----
    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "p10": round(float(np.percentile(arr, 10)), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "median": round(float(np.median(arr)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
        }

    summary = {"per_judgment": {}, "boundaries": {}}
    by_judgment = defaultdict(list)
    for row in rows:
        by_judgment[row["judgment"]].append(row)
    for judgment, items in sorted(by_judgment.items(), key=lambda kv: -len(kv[1])):
        summary["per_judgment"][judgment] = {
            feature: stats([it[feature] for it in items])
            for feature in ("r_contact", "r_minor", "obj_aspect", "contact_thick", "hand_palm")
        }

    def best_threshold(feature, low_group, high_group):
        """low_group < t <= high_group 가정에서 정확도 최대 임계값."""

        low = [it[feature] for it in rows if it["judgment"] in low_group]
        high = [it[feature] for it in rows if it["judgment"] in high_group]
        if not low or not high:
            return None
        values = sorted(set(low + high))
        best = None
        for t in values:
            correct = sum(1 for v in low if v <= t) + sum(1 for v in high if v > t)
            accuracy = correct / (len(low) + len(high))
            if best is None or accuracy > best["accuracy"]:
                best = {"threshold": round(t, 3), "accuracy": round(accuracy, 3),
                        "n_low": len(low), "n_high": len(high)}
        return best

    # 집기(rim_pinch) vs 감기(handle 파지): PRECISION/WRAP 경계에 해당
    for feature in ("r_contact", "r_minor", "contact_thick"):
        summary["boundaries"][f"pinch_vs_handle::{feature}"] = best_threshold(
            feature, {"rim_pinch"}, {"handle"})
    # 감기 vs 몸통(POWER 참고용)
    for feature in ("r_contact", "r_minor"):
        summary["boundaries"][f"handle_vs_body::{feature}"] = best_threshold(
            feature, {"handle"}, {"body"})

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {summary_path}")
    print(json.dumps(summary["boundaries"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
