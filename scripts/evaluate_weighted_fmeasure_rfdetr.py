#!/usr/bin/env python
"""RF-DETR-Seg 체크포인트로 UMD test의 weighted F-measure(F_beta^w)를 계산한다.

`scripts/evaluate_weighted_fmeasure.py`(Ultralytics 전용)와 같은 지표·같은 GT를
쓰되 추론만 RF-DETR로 바꾼 판이다. 최종 채택 모델이 RF-DETR 계열이므로 선행 연구
(AffordanceNet 등, UMD 평균 0.799)와 같은 잣대로 비교하려면 이 경로가 필요하다.

정의: Margolin et al., "How to Evaluate Foreground Maps", CVPR 2014.

주의(공정 비교):
- UMD 원 정의는 grasp = grasp + wrap-grasp이므로 `--merge-grasp`를 주면 handle/body
  확률을 픽셀별 최댓값으로 합쳐 grasp_region 하나로 평가한다.
- functional_region은 UMD의 cut/scoop/pound/support를 합친 우리 정의를 그대로 쓴다.
- contain은 정책상 ignore이므로 GT·평가에서 제외한다.
- RF-DETR은 인스턴스별 이진 마스크를 주므로 신뢰도를 곱해 소프트 확률 맵으로 만든다.
  Ultralytics 판(마스크 확률 × 신뢰도)과 같은 방식이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_weighted_fmeasure import weighted_f_measure  # noqa: E402

# 모델 클래스 ID (yolo_grasp_type_* 정의와 동일)
CLASS_IDS = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}


def rasterize_ground_truth(label_path: Path, height: int, width: int,
                           eval_classes: dict[str, set[int]]) -> dict[str, np.ndarray]:
    """YOLO 폴리곤 라벨을 클래스별 합집합 이진 마스크로 만든다."""

    gt = {name: np.zeros((height, width), dtype=bool) for name in eval_classes}
    if not label_path.is_file():
        return gt
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        cls = int(fields[0])
        pts = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        for name, ids in eval_classes.items():
            if cls in ids:
                canvas = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(canvas, [pts.astype(np.int32)], 1)
                gt[name] |= canvas.astype(bool)
    return gt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="RF-DETR 체크포인트(.pth)")
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--data", type=Path, required=True, help="3클래스 dataset.yaml (test 분할 사용)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="낮게 두어 소프트 맵을 충분히 채운다 (F-measure는 확률 맵 기반)")
    parser.add_argument("--merge-grasp", action="store_true",
                        help="handle+body를 grasp 하나로 합쳐 UMD 원 정의에 맞춘다")
    args = parser.parse_args()

    import yaml
    from PIL import Image
    import rfdetr

    document = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    test_dir = (args.data.parent / str(document["test"])).resolve()
    label_dir = args.data.parent / "labels" / test_dir.name
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise SystemExit(f"test 이미지를 찾지 못했습니다: {test_dir}")

    if not hasattr(rfdetr, args.variant):
        raise SystemExit(f"rfdetr에 없는 모델 변형입니다: {args.variant}")
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(args.weights.resolve()))

    if args.merge_grasp:
        eval_classes: dict[str, set[int]] = {"grasp_region": {0, 1}, "functional_region": {2}}
    else:
        eval_classes = {"handle_grasp_region": {0}, "body_grasp_region": {1},
                        "functional_region": {2}}

    scores: dict[str, list[float]] = {name: [] for name in eval_classes}
    class_id_base: int | None = None

    for index, image_path in enumerate(images, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        gt = rasterize_ground_truth(label_dir / f"{image_path.stem}.txt", height, width, eval_classes)
        if not any(mask.any() for mask in gt.values()):
            continue

        prob = {name: np.zeros((height, width), dtype=np.float64) for name in eval_classes}
        detections = model.predict(image, threshold=args.threshold)
        if detections.mask is not None and len(detections) > 0:
            raw_ids = np.asarray(detections.class_id, dtype=int)
            # class id가 0시작인지 1시작(COCO category)인지 첫 관측으로 확정한다.
            if class_id_base is None:
                class_id_base = 1 if raw_ids.max(initial=0) >= len(CLASS_IDS) else 0
            confidences = np.asarray(detections.confidence, dtype=float)
            for mask, raw_id, conf in zip(np.asarray(detections.mask), raw_ids, confidences):
                cls = int(raw_id) - class_id_base
                if cls not in CLASS_IDS:
                    continue
                mask_f = mask.astype(np.float64)
                if mask_f.shape != (height, width):
                    mask_f = cv2.resize(mask_f, (width, height), interpolation=cv2.INTER_LINEAR)
                for name, ids in eval_classes.items():
                    if cls in ids:
                        prob[name] = np.maximum(prob[name], mask_f * float(conf))

        for name in eval_classes:
            if gt[name].any():
                scores[name].append(weighted_f_measure(np.clip(prob[name], 0, 1), gt[name]))

        if index % 200 == 0:
            print(f"  {index}/{len(images)} 처리")

    summary = {
        "model": str(args.weights),
        "variant": args.variant,
        "backend": "rfdetr",
        "test_images": len(images),
        "merge_grasp": args.merge_grasp,
        "prediction_threshold": args.threshold,
        "rfdetr_class_id_base": class_id_base,
        "note": "UMD test 전용. contain=ignore 제외. 클래스 정의가 문헌과 완전 동일하지 않음.",
        "per_class_weighted_fmeasure": {},
    }
    means = []
    for name, values in scores.items():
        m = float(np.nanmean(values)) if values else None
        summary["per_class_weighted_fmeasure"][name] = {
            "mean": round(m, 4) if m is not None else None,
            "gt_present_images": len(values),
        }
        if m is not None:
            means.append(m)
    summary["average_weighted_fmeasure"] = round(float(np.mean(means)), 4) if means else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
