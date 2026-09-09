#!/usr/bin/env python
"""robot_hand 클래스용 손 마스크 후보를 생성한다 (Grounding DINO + SAM2).

data/raw/custom/hand_jetson_frames 의 프레임에서 로봇손 상자를 제로샷 검출하고
SAM2로 마스크를 만든다. 결과는 사람이 검수하기 전에는 정답으로 쓰지 않는다
(머그 의사라벨 파이프라인과 같은 원칙).

프레임에는 로봇손을 든 사람 손이 함께 나오므로, 프롬프트는 검은 로봇손을
특정하는 문구를 우선 쓰고 실패 시 대체 문구를 시도한다. 잘못 잡힌 프레임은
검수에서 제외하면 된다.

출력 (output-dir 아래):
  masks/<영상>/<이름>.png      0/255 손 마스크
  overlays/<영상>/<이름>.jpg   검수용 오버레이 (초록 윤곽 + 상자 + 점수)
  candidates.jsonl             프레임별 결과 기록 (프롬프트, 점수, 상태)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/raw/custom/hand_jetson_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/hand_pseudolabels"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cache-dir", type=Path, default=None,
                        help="모델 가중치 캐시. 서버에서는 반드시 추가 디스크 경로 사용")
    parser.add_argument("--detector-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-px", type=int, default=1500,
                        help="이보다 작은 마스크는 손이 아니라고 보고 버린다")
    return parser.parse_args()


PROMPTS = [
    "a black robotic hand",
    "a black prosthetic hand",
    "a robot hand",
]


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )
    import cv2

    cache = str(args.model_cache_dir) if args.model_cache_dir else None
    detector_processor = AutoProcessor.from_pretrained(args.detector_id, cache_dir=cache)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.detector_id, cache_dir=cache
    ).to(args.device)
    sam2_processor = Sam2Processor.from_pretrained(args.sam2_id, cache_dir=cache)
    sam2 = Sam2Model.from_pretrained(args.sam2_id, cache_dir=cache).to(args.device)

    frames = sorted(args.frames_dir.rglob("*.jpg"))
    if not frames:
        raise SystemExit(f"프레임이 없습니다: {args.frames_dir}")
    print(f"프레임 {len(frames)}장, 장치 {args.device}")

    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    records = []

    def detect(image: Image.Image, prompt: str):
        inputs = detector_processor(images=image, text=[[prompt]], return_tensors="pt").to(args.device)
        with torch.inference_mode():
            outputs = detector(**inputs)
        result = detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        best = None
        for box, score in zip(result["boxes"], result["scores"]):
            score = float(score)
            if best is None or score > best[1]:
                best = ([float(v) for v in box.tolist()], score)
        return best

    def segment(image: Image.Image, box):
        inputs = sam2_processor(
            images=image, input_boxes=[[box]], return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            outputs = sam2(**inputs, multimask_output=True)
        masks = sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0][0]
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        best_index = int(scores.argmax())
        return masks[best_index].numpy() > 0.5, float(scores[best_index])

    detected = 0
    for i, path in enumerate(frames, 1):
        relative = path.relative_to(args.frames_dir)
        image = Image.open(path).convert("RGB")
        record = {"image": str(relative).replace("\\", "/"), "status": "no_detection",
                  "prompt": None, "box_score": None, "mask_iou": None, "mask_px": 0}

        found = None
        for prompt in PROMPTS:
            best = detect(image, prompt)
            if best is not None:
                found = (prompt, best)
                break
        if found is not None:
            prompt, (box, box_score) = found
            mask, mask_iou = segment(image, box)
            mask_px = int(mask.sum())
            if mask_px >= args.min_mask_px:
                record.update({"status": "ok", "prompt": prompt,
                               "box_score": round(box_score, 3),
                               "mask_iou": round(mask_iou, 3), "mask_px": mask_px,
                               "box": [round(v, 1) for v in box]})
                mask_path = args.output_dir / "masks" / relative.with_suffix(".png")
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                cv2.putText(frame, f"{prompt} {box_score:.2f}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                overlay_path = args.output_dir / "overlays" / relative
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), frame)
                detected += 1
            else:
                record["status"] = "mask_too_small"
                record["mask_px"] = mask_px
        records.append(record)
        if i % 25 == 0:
            print(f"  {i}/{len(frames)} 처리, 검출 {detected}")

    with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"완료: {detected}/{len(frames)} 검출. 출력: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
