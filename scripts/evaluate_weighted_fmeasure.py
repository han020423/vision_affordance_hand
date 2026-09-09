#!/usr/bin/env python
"""UMD test에서 weighted F-measure(F_beta^w)를 계산한다.

부품 affordance 분할 문헌(AffordanceNet 등)의 표준 지표로, mAP/IoU와 별개로
AffordanceNet(UMD 평균 0.799)과 같은 잣대의 비교값을 얻기 위한 계산이다.

정의: Margolin et al., "How to Evaluate Foreground Maps", CVPR 2014.
정답 근처 오류는 의존적으로 묶고(의존성 가중), 정답에서 먼 오검출일수록 크게
벌한다(위치 가중). 입력은 예측 소프트 확률 맵과 이진 정답이다.

주의(공정 비교):
- UMD 원 정의는 grasp = grasp + wrap-grasp이므로, 모델의 handle/body 확률을
  픽셀별 최댓값으로 합쳐 grasp_region 하나로 평가한다(--merge-grasp).
- functional_region은 UMD의 cut/scoop/pound/support를 합친 우리 정의를 그대로
  쓴다. 클래스 정의가 문헌과 완전히 동일하지 않음을 결과에 명시한다.
- contain은 정책상 ignore이므로 GT·평가에서 제외한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


def weighted_f_measure(prediction: np.ndarray, ground_truth: np.ndarray, beta_sq: float = 0.3) -> float:
    """Margolin(2014) weighted F-measure. prediction은 [0,1], GT는 bool."""

    gt = ground_truth.astype(bool)
    if not gt.any():
        return float("nan")
    dst, idx = distance_transform_edt(~gt, return_indices=True)
    # E: 절대 오차. 정답 밖 픽셀의 오차를 가장 가까운 정답 픽셀의 오차로 대체(의존성).
    e = np.abs(prediction - gt.astype(np.float64))
    et = e.copy()
    outside = ~gt
    et[outside] = e[idx[0][outside], idx[1][outside]]
    # A: 정답에서 멀수록 큰 가중(가우시안 감쇠).
    sigma = 5.0
    ea = gaussian_filter(et, sigma=sigma, truncate=3.0, mode="constant")
    min_e_ea = np.where((e < ea) & gt, ea, e)
    b = np.where(outside, 2.0 - np.exp(np.log(0.5) / 5.0 * dst), np.ones_like(dst))
    ew = min_e_ea * b

    tp_w = (1.0 - ew)[gt].sum()
    fp_w = ew[outside].sum()
    precision = tp_w / (tp_w + fp_w + 1e-12)
    recall = 1.0 - ew[gt].mean()
    denom = beta_sq * precision + recall
    if denom < 1e-12:
        return 0.0
    return float((1 + beta_sq) * precision * recall / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="3클래스 dataset.yaml (test 분할 사용)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--merge-grasp", action="store_true",
                        help="handle+body를 grasp 하나로 합쳐 UMD 원 정의에 맞춘다")
    args = parser.parse_args()

    import yaml

    document = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    test_dir = (args.data.parent / str(document["test"])).resolve()
    label_dir = args.data.parent / "labels" / test_dir.name
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPOSITORY_ROOT / ".ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))

    # 평가 클래스 정의
    if args.merge_grasp:
        eval_classes = {"grasp_region": {0, 1}, "functional_region": {2}}
    else:
        eval_classes = {"handle_grasp_region": {0}, "body_grasp_region": {1}, "functional_region": {2}}

    scores: dict[str, list[float]] = {name: [] for name in eval_classes}
    predictions = model.predict(source=str(test_dir), imgsz=args.imgsz, conf=0.001,
                                iou=0.7, device=args.device, retina_masks=True,
                                stream=True, verbose=False)
    for result in predictions:
        image_path = Path(result.path)
        height, width = result.orig_shape
        # GT 래스터화 (클래스별 합집합)
        gt = {name: np.zeros((height, width), dtype=bool) for name in eval_classes}
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
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
        # 예측 소프트 확률 맵 (클래스별, 인스턴스 신뢰도로 가중한 최댓값)
        prob = {name: np.zeros((height, width), dtype=np.float64) for name in eval_classes}
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            for mask, cls, conf in zip(masks, classes, confs):
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                for name, ids in eval_classes.items():
                    if cls in ids:
                        prob[name] = np.maximum(prob[name], mask.astype(np.float64) * float(conf))
        for name in eval_classes:
            if gt[name].any():
                scores[name].append(weighted_f_measure(np.clip(prob[name], 0, 1), gt[name]))

    result_summary = {"model": str(args.model), "test_images": len(images),
                      "merge_grasp": args.merge_grasp,
                      "note": "UMD test 전용. contain=ignore 제외. 클래스 정의가 문헌과 완전 동일하지 않음.",
                      "per_class_weighted_fmeasure": {}, }
    all_means = []
    for name, values in scores.items():
        m = float(np.nanmean(values)) if values else None
        result_summary["per_class_weighted_fmeasure"][name] = {
            "mean": round(m, 4) if m is not None else None,
            "gt_present_images": len(values),
        }
        if m is not None:
            all_means.append(m)
    result_summary["average_weighted_fmeasure"] = round(float(np.mean(all_means)), 4) if all_means else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
