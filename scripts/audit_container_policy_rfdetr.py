#!/usr/bin/env python
"""RF-DETR 모델의 contain 정책 감사: ignore(255) 픽셀 위 functional 오예측 검사.

evaluate_grasp_type_model.py의 policy_audit와 같은 프로토콜을 RF-DETR 백엔드로
수행한다. 감사 묶음(manifest.jsonl)의 UMD 보류 용기 117장(cup_03, mug_20, pot_02)에
대해, 고정 신뢰도에서 예측한 functional_region이 semantic 255(contain=ignore)
픽셀과 겹치는 이미지 수를 센다. 목표는 0장이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONAL_CLASS_ID = 2  # 0=handle, 1=body, 2=functional (0시작, 평가에서 확인됨)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True, help="감사 결과 JSON 경로")
    args = parser.parse_args()

    import cv2
    from PIL import Image
    import rfdetr

    manifest_path = args.policy_manifest.resolve()
    output_path = args.output if args.output.is_absolute() else (REPOSITORY_ROOT / args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"기존 감사 결과를 덮어쓰지 않습니다: {output_path}")

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    with manifest_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if not hasattr(rfdetr, args.variant):
        raise ValueError(f"rfdetr에 없는 모델 변형입니다: {args.variant}")
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(args.model.resolve()))

    rows: list[dict[str, object]] = []
    images_with_violation = 0
    for source in records:
        semantic = cv2.imread(str(resolve(source["semantic_mask_path"])), cv2.IMREAD_GRAYSCALE)
        if semantic is None:
            raise OSError(f"semantic 마스크를 읽지 못했습니다: {source['source_id']}")
        height, width = semantic.shape
        detections = model.predict(
            Image.open(resolve(source["image_path"])).convert("RGB"), threshold=args.confidence
        )
        functional_pred = np.zeros((height, width), dtype=bool)
        if detections is not None and detections.mask is not None and len(detections) > 0:
            for mask, class_id in zip(
                np.asarray(detections.mask), np.asarray(detections.class_id, dtype=int)
            ):
                if int(class_id) != FUNCTIONAL_CLASS_ID:
                    continue
                mask = mask.astype(np.uint8)
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                functional_pred |= mask.astype(bool)
        ignore = semantic == 255
        overlap = int(np.count_nonzero(functional_pred & ignore))
        ignore_pixels = int(np.count_nonzero(ignore))
        if overlap > 0:
            images_with_violation += 1
        rows.append(
            {
                "source_id": source["source_id"],
                "object_id": source.get("object_id"),
                "functional_on_ignore_pixels": overlap,
                "ignore_covered_ratio": overlap / ignore_pixels if ignore_pixels else None,
            }
        )

    summary = {
        "evaluation_kind": "separate_container_policy_audit_not_formal_test_metrics",
        "policy": "UMD contain은 ignore로 유지하며 컵·용기 몸통을 functional_region으로 바꾸지 않습니다.",
        "backend": "rfdetr",
        "variant": args.variant,
        "model": str(args.model.resolve()),
        "prediction_confidence": args.confidence,
        "image_count": len(rows),
        "images_with_functional_on_contain": images_with_violation,
        "per_image": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
