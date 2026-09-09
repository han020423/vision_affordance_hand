#!/usr/bin/env python
"""RF-DETR-Seg 체크포인트의 mask mAP를 YOLO 계보와 같은 test·같은 잣대로 계산한다.

배경: `scripts/evaluate_grasp_type_model.py`는 Ultralytics 전용이라 RF-DETR 계열의
mAP를 낼 수 없었다. 그래서 계열 비교를 픽셀 IoU로만 해 왔는데, 발표·보고서에서
YOLO 계보의 mAP 표와 나란히 놓으려면 같은 GT·같은 지표가 필요하다.

방식: YOLO 폴리곤 라벨(test 분할)을 COCO 형식 GT로 변환하고, RF-DETR 예측을 COCO
결과 형식으로 만들어 pycocotools의 segm 평가를 돌린다. Ultralytics의 mAP도 동일한
COCO 정의(IoU 0.50:0.95)를 따르므로 값이 서로 대응된다.

주의:
- Ultralytics는 자체 구현으로 AP를 계산하므로 pycocotools 값과 소수점 아래에서
  미세한 차이가 날 수 있다. 표에 함께 실을 때 이 점을 명시한다.
- 클래스는 0=handle, 1=body, 2=functional이며 robot_hand(3)가 있는 모델은 무시한다.
- 신뢰도 임계값은 낮게 두어야 AP가 정상 계산된다(재현율 곡선 전체가 필요).
"""

from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

CLASS_NAMES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}


def build_coco_ground_truth(images: list[Path], label_dir: Path) -> dict:
    """YOLO 폴리곤 라벨을 COCO GT(dict)로 변환한다. 마스크는 폴리곤 그대로 쓴다."""

    coco = {"images": [], "annotations": [],
            "categories": [{"id": cid, "name": name} for cid, name in CLASS_NAMES.items()]}
    ann_id = 1
    for image_id, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        coco["images"].append({"id": image_id, "file_name": image_path.name,
                               "height": height, "width": width})
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 7:
                continue
            cls = int(fields[0])
            if cls not in CLASS_NAMES:
                continue
            pts = np.asarray([float(v) for v in fields[1:]], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] *= width
            pts[:, 1] *= height
            xs, ys = pts[:, 0], pts[:, 1]
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            area = float(cv2.contourArea(pts.astype(np.float32)))
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": cls,
                "segmentation": [pts.reshape(-1).tolist()],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": max(area, 1.0), "iscrowd": 0,
            })
            ann_id += 1
    return coco


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--data", type=Path, required=True, help="3클래스 dataset.yaml (test 분할)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="AP는 재현율 곡선 전체가 필요하므로 낮게 둔다")
    parser.add_argument("--class-id-base", type=int, choices=[0, 1], default=0,
                        help="예측 class id 시작값. 우리 모델은 0시작이다. "
                             "자동 추정은 낮은 임계값에서 robot_hand(3) 잡음에 속으므로 명시한다")
    args = parser.parse_args()

    import yaml
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util
    import rfdetr

    document = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    test_dir = (args.data.parent / str(document["test"])).resolve()
    label_dir = args.data.parent / "labels" / test_dir.name
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise SystemExit(f"test 이미지를 찾지 못했습니다: {test_dir}")
    print(f"test 이미지 {len(images)}장, GT 변환 중...")

    gt_dict = build_coco_ground_truth(images, label_dir)
    print(f"GT 인스턴스 {len(gt_dict['annotations'])}개")

    if not hasattr(rfdetr, args.variant):
        raise SystemExit(f"rfdetr에 없는 모델 변형입니다: {args.variant}")
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(args.weights.resolve()))

    results: list[dict] = []
    class_id_base = args.class_id_base
    for entry, image_path in zip(gt_dict["images"], images):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        detections = model.predict(image, threshold=args.threshold)
        if detections.mask is None or len(detections) == 0:
            continue
        raw_ids = np.asarray(detections.class_id, dtype=int)
        confidences = np.asarray(detections.confidence, dtype=float)
        for mask, raw_id, conf in zip(np.asarray(detections.mask), raw_ids, confidences):
            cls = int(raw_id) - class_id_base
            if cls not in CLASS_NAMES:
                continue
            binary = mask.astype(np.uint8)
            if binary.shape != (height, width):
                binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
            if not binary.any():
                continue
            rle = mask_util.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({"image_id": entry["id"], "category_id": cls,
                            "segmentation": rle, "score": float(conf)})
        if entry["id"] % 200 == 0:
            print(f"  {entry['id']}/{len(images)} 추론")

    print(f"예측 인스턴스 {len(results)}개, COCO 평가 시작")
    with contextlib.redirect_stdout(_io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results) if results else None

    summary: dict = {"model": str(args.weights), "variant": args.variant, "backend": "rfdetr",
                     "test_images": len(images), "prediction_threshold": args.threshold,
                     "rfdetr_class_id_base": class_id_base,
                     "note": "GT는 YOLO 폴리곤 라벨을 COCO로 변환. Ultralytics AP 구현과 소수점 아래 차이 가능.",
                     "masks": {}}
    if coco_dt is None:
        raise SystemExit("예측이 없어 평가할 수 없습니다.")

    def run_eval(cat_ids=None):
        with contextlib.redirect_stdout(_io.StringIO()):
            e = COCOeval(coco_gt, coco_dt, iouType="segm")
            if cat_ids is not None:
                e.params.catIds = cat_ids
            e.evaluate(); e.accumulate(); e.summarize()
        return e.stats

    stats = run_eval()
    summary["masks"]["overall"] = {"map50_95": round(float(stats[0]), 4),
                                   "map50": round(float(stats[1]), 4),
                                   "map75": round(float(stats[2]), 4)}
    summary["masks"]["per_class"] = {}
    for cid, name in CLASS_NAMES.items():
        s = run_eval([cid])
        summary["masks"]["per_class"][name] = {"map50_95": round(float(s[0]), 4),
                                               "map50": round(float(s[1]), 4)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
