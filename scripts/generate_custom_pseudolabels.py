#!/usr/bin/env python
"""Grounding DINO와 SAM2로 자체 이미지의 검수용 마스크 후보를 생성한다."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.common import save_mask  # noqa: E402
from src.labeling.custom_pseudolabels import (  # noqa: E402
    MaskCandidate,
    combine_category_candidates,
    combine_grasp_candidates,
    resolve_repository_path,
    select_even_samples,
    split_anchor_residual,
    split_scissors_by_rings,
    split_screwdriver_by_width,
    split_whole_mug_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grounding DINO와 SAM2로 머그 손잡이·몸통 마스크 후보를 만들되, "
            "사람이 검수하기 전에는 정답으로 확정하지 않습니다."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/custom_pseudolabels.yaml"),
        help="모델, 프롬프트, 임계값과 출력 경로를 기록한 YAML",
    )
    parser.add_argument(
        "--sample-per-object",
        type=int,
        help="각 실제 물체에서 시간순으로 고르게 뽑을 장수. 생략하면 전체 처리",
    )
    parser.add_argument(
        "--run-name",
        default="full_candidates",
        help="기존 결과와 섞이지 않도록 사용할 실행 폴더 이름",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="추론 장치. 예: cuda:0 또는 cpu",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="모델 가중치 캐시 경로. 서버에서는 반드시 추가 디스크 경로 사용",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="처리할 object_id 목록. 생략하면 pending_annotation 전체를 처리",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest JSON 오류: {path}:{line_number}") from exc
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    policy = config.get("policy") or {}
    classes = {int(key): value for key, value in (policy.get("classes") or {}).items()}
    if classes != {0: "grasp_region", 1: "functional_region"}:
        raise ValueError(f"고정 클래스 정책과 다릅니다: {classes}")
    if int(policy.get("ignore_index", -1)) != 255:
        raise ValueError("ignore_index는 255여야 합니다")
    if not bool(policy.get("mug_body_is_not_functional_region", False)):
        raise ValueError("컵 몸통을 functional_region으로 만들 수 없습니다")
    if not bool(policy.get("separate_grasp_components", False)):
        raise ValueError("손잡이와 몸통은 별도 인스턴스로 유지해야 합니다")
    if not bool(policy.get("require_human_review", False)):
        raise ValueError("반자동 후보는 반드시 사람 검수 상태여야 합니다")

    # v2: 카테고리별 프롬프트·부위 정책. functional 잔여 부위는 승인된 카테고리에만 허용한다.
    categories = config.get("categories") or {}
    allowed_functional = set(policy.get("functional_allowed_categories") or [])
    for name, entry in categories.items():
        method = str(entry.get("method", ""))
        if method not in (
            "whole_geometric_mug",
            "anchor_residual",
            "scissors_holes",
            "screwdriver_width",
        ):
            raise ValueError(f"알 수 없는 카테고리 방식입니다({name}): {method}")
        if "whole_prompt" not in entry:
            raise ValueError(f"카테고리 {name}에 whole_prompt가 없습니다")
        if method != "whole_geometric_mug":
            required = ["anchor_part", "residual_part", "residual_label"]
            if method == "anchor_residual":
                required.append("anchor_prompt")
            for key in required:
                if key not in entry:
                    raise ValueError(f"카테고리 {name}에 {key}가 없습니다")
            if entry["residual_label"] == "functional_region" and name not in allowed_functional:
                raise ValueError(
                    f"카테고리 {name}은 functional 생성이 승인되지 않았습니다"
                )
    return config


def _part_from_label(label: object) -> str | None:
    """Grounding DINO 반환 문구가 컵 전체 후보인지 확인한다."""

    normalized = str(label).lower()
    if "mug" in normalized or "cup" in normalized or normalized.isdigit():
        return "whole"
    return None


def _detect_best_boxes(
    image: Image.Image,
    prompts: dict[str, str],
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
    torch: Any,
) -> list[dict[str, object]]:
    """컵 전체를 가리키는 최고 점수 상자 하나만 남긴다."""

    text_labels = [[prompts["whole"]]]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    best: dict[str, dict[str, object]] = {}
    for box, score, label in zip(
        result["boxes"], result["scores"], result["labels"], strict=True
    ):
        part = _part_from_label(label)
        if part is None:
            continue
        row = {
            "part": part,
            "score": float(score.detach().cpu().item()),
            "box_xyxy": tuple(float(value) for value in box.detach().cpu().tolist()),
            "detector_label": str(label),
        }
        if part not in best or float(row["score"]) > float(best[part]["score"]):
            best[part] = row
    return [best["whole"]] if "whole" in best else []


def _detect_prompt_boxes(
    image: Image.Image,
    prompt: str,
    max_boxes: int,
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
    torch: Any,
) -> list[dict[str, object]]:
    """단일 프롬프트로 최고 점수 상자를 최대 max_boxes개 반환한다.

    v2 카테고리 경로용이다. 프롬프트가 하나뿐이므로 라벨 문구 대조 없이
    점수 순으로 자르며, 상자 간 판정은 사람 검수 단계에 맡긴다.
    """

    inputs = processor(images=image, text=[[prompt]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    rows = [
        {
            "score": float(score.detach().cpu().item()),
            "box_xyxy": tuple(float(value) for value in box.detach().cpu().tolist()),
            "detector_label": str(label),
        }
        for box, score, label in zip(
            result["boxes"], result["scores"], result["labels"], strict=True
        )
    ]
    rows.sort(key=lambda row: -float(row["score"]))
    return rows[:max_boxes]


def _segment_boxes(
    image: Image.Image,
    detections: list[dict[str, object]],
    processor: Any,
    model: Any,
    device: str,
    torch: Any,
) -> list[MaskCandidate]:
    if not detections:
        return []
    boxes = [[list(detection["box_xyxy"]) for detection in detections]]
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
    )[0]
    iou_scores = outputs.iou_scores.detach().cpu()[0]
    candidates: list[MaskCandidate] = []
    for index, detection in enumerate(detections):
        best_mask_index = int(torch.argmax(iou_scores[index]).item())
        mask = masks[index, best_mask_index].numpy() > 0
        candidates.append(
            MaskCandidate(
                part=str(detection["part"]),
                score=float(detection["score"]),
                box_xyxy=tuple(detection["box_xyxy"]),
                mask=mask,
            )
        )
    return candidates


def _render_overlay(
    image: Image.Image,
    instance: np.ndarray,
    components: list[dict[str, object]],
) -> Image.Image:
    """손잡이(초록)와 몸통(청록)을 구분한 검수 이미지를 만든다."""

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    color_by_part = {
        "handle": np.array([45, 220, 90], dtype=np.float32),
        "body": np.array([20, 190, 230], dtype=np.float32),
        # 기능 부위(가위 날, 드라이버 축)는 빨간 계열로 표시한다.
        "blade": np.array([235, 60, 60], dtype=np.float32),
        "shaft": np.array([235, 60, 60], dtype=np.float32),
    }
    fallback_by_label = {
        "grasp_region": np.array([45, 220, 90], dtype=np.float32),
        "functional_region": np.array([235, 60, 60], dtype=np.float32),
    }
    for component in components:
        instance_id = int(component["instance_id"])
        color = color_by_part.get(
            str(component["part"]),
            fallback_by_label[str(component.get("label", "grasp_region"))],
        )
        mask = instance == instance_id
        overlay[mask] = base[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    config_path = resolve_repository_path(REPOSITORY_ROOT, args.config)
    config = _load_config(config_path)
    manifest_path = resolve_repository_path(REPOSITORY_ROOT, config["manifest_path"])
    output_base = resolve_repository_path(REPOSITORY_ROOT, config["output_root"])
    output_root = output_base / args.run_name
    if output_root.exists():
        raise FileExistsError(f"기존 반자동 결과를 덮어쓰지 않습니다: {output_root}")
    if args.model_cache_dir is not None and not args.model_cache_dir.is_absolute():
        raise ValueError("모델 캐시는 서버 추가 디스크의 절대경로로 지정하세요")

    all_records = _load_jsonl(manifest_path)
    if args.objects:
        wanted = set(args.objects)
        known = {str(row["object_id"]) for row in all_records}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"manifest에 없는 object_id입니다: {sorted(unknown)}")
        all_records = [row for row in all_records if str(row["object_id"]) in wanted]
    records = select_even_samples(all_records, args.sample_per_object)
    if not records:
        raise ValueError("처리할 pending_annotation 자체 이미지가 없습니다")

    # 무거운 의존성은 실행 시점에만 불러와 데이터 단위 테스트와 분리한다.
    import torch
    import transformers
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 요청했지만 PyTorch가 GPU를 찾지 못했습니다")

    cache_dir = str(args.model_cache_dir) if args.model_cache_dir else None
    model_config = config["models"]
    detector_id = str(model_config["grounding_dino"])
    sam2_id = str(model_config["sam2"])
    detector_processor = AutoProcessor.from_pretrained(detector_id, cache_dir=cache_dir)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_id, cache_dir=cache_dir
    ).to(args.device)
    sam2_processor = Sam2Processor.from_pretrained(sam2_id, cache_dir=cache_dir)
    sam2 = Sam2Model.from_pretrained(sam2_id, cache_dir=cache_dir).to(args.device)
    detector.eval()
    sam2.eval()

    thresholds = config["thresholds"]
    prompts = {key: str(value) for key, value in (config.get("prompts") or {}).items()}
    categories: dict[str, dict[str, Any]] = config.get("categories") or {}
    staging = output_base / f".{args.run_name}.tmp"
    if staging.exists():
        raise FileExistsError(f"이전 중단 결과를 먼저 확인하세요: {staging}")
    rows: list[dict[str, object]] = []
    try:
        for number, record in enumerate(records, start=1):
            source_id = str(record["source_id"])
            image_path = resolve_repository_path(REPOSITORY_ROOT, record["image_path"])
            image = Image.open(image_path).convert("RGB")
            minimum_pixels = int(thresholds["minimum_mask_pixels"])
            category = str(record.get("category", ""))
            category_config = categories.get(category)
            shape_diagnostics: dict[str, object] = {}

            if category_config is None:
                # v1 경로: 머그 전용 설정(prompts.whole)과 기하 분리를 그대로 사용한다.
                detections = _detect_best_boxes(
                    image,
                    prompts,
                    detector_processor,
                    detector,
                    args.device,
                    float(thresholds["box"]),
                    float(thresholds["text"]),
                    torch,
                )
                candidates = _segment_boxes(
                    image, detections, sam2_processor, sam2, args.device, torch
                )
                if candidates:
                    whole = candidates[0]
                    body_mask, handle_mask, shape_flags, shape_diagnostics = (
                        split_whole_mug_mask(
                            whole.mask, minimum_handle_pixels=minimum_pixels
                        )
                    )
                    separated_candidates = [
                        MaskCandidate("handle", whole.score, whole.box_xyxy, handle_mask),
                        MaskCandidate("body", whole.score, whole.box_xyxy, body_mask),
                    ]
                    semantic, instance, components, combine_flags = combine_grasp_candidates(
                        separated_candidates,
                        (image.height, image.width),
                        minimum_pixels=minimum_pixels,
                    )
                    flags = sorted(set(shape_flags + combine_flags))
                else:
                    semantic, instance, components, flags = combine_grasp_candidates(
                        [], (image.height, image.width), minimum_pixels=minimum_pixels
                    )
                    flags = sorted(set(flags + ["컵_전체_탐지_누락"]))
            else:
                # v2 경로: 카테고리 설정으로 전체·앵커 부위를 탐지한다.
                whole_boxes = _detect_prompt_boxes(
                    image,
                    str(category_config["whole_prompt"]),
                    1,
                    detector_processor,
                    detector,
                    args.device,
                    float(thresholds["box"]),
                    float(thresholds["text"]),
                    torch,
                )
                whole_candidates = _segment_boxes(
                    image,
                    [{"part": "whole", **box} for box in whole_boxes],
                    sam2_processor,
                    sam2,
                    args.device,
                    torch,
                )
                if not whole_candidates:
                    semantic = np.zeros((image.height, image.width), dtype=np.uint8)
                    instance = np.zeros((image.height, image.width), dtype=np.uint16)
                    components = []
                    flags = ["유효한_마스크_없음", "전체_탐지_누락"]
                elif str(category_config["method"]) == "whole_geometric_mug":
                    whole = whole_candidates[0]
                    body_mask, handle_mask, shape_flags, shape_diagnostics = (
                        split_whole_mug_mask(
                            whole.mask, minimum_handle_pixels=minimum_pixels
                        )
                    )
                    separated_candidates = [
                        MaskCandidate("handle", whole.score, whole.box_xyxy, handle_mask),
                        MaskCandidate("body", whole.score, whole.box_xyxy, body_mask),
                    ]
                    semantic, instance, components, combine_flags = combine_grasp_candidates(
                        separated_candidates,
                        (image.height, image.width),
                        minimum_pixels=minimum_pixels,
                    )
                    flags = sorted(set(shape_flags + combine_flags))
                else:
                    whole = whole_candidates[0]
                    method = str(category_config["method"])
                    if method == "scissors_holes":
                        # 구멍(고리) 기반 기하 분리. DINO 부위 프롬프트가 전체를
                        # 반환하는 문제(2026-08-21) 때문에 프롬프트 분리를 쓰지 않는다.
                        anchor_components, residual_mask, split_flags, shape_diagnostics = (
                            split_scissors_by_rings(
                                whole.mask, minimum_pixels=minimum_pixels
                            )
                        )
                    elif method == "screwdriver_width":
                        anchor_components, residual_mask, split_flags, shape_diagnostics = (
                            split_screwdriver_by_width(
                                whole.mask, minimum_pixels=minimum_pixels
                            )
                        )
                    else:  # anchor_residual: 부위 프롬프트 기반(현재 기본 미사용)
                        max_instances = int(category_config.get("anchor_max_instances", 1))
                        anchor_boxes = _detect_prompt_boxes(
                            image,
                            str(category_config["anchor_prompt"]),
                            max_instances,
                            detector_processor,
                            detector,
                            args.device,
                            float(thresholds["box"]),
                            float(thresholds["text"]),
                            torch,
                        )
                        anchor_candidates = _segment_boxes(
                            image,
                            [
                                {"part": str(category_config["anchor_part"]), **box}
                                for box in anchor_boxes
                            ],
                            sam2_processor,
                            sam2,
                            args.device,
                            torch,
                        )
                        anchor_mask = np.zeros((image.height, image.width), dtype=bool)
                        for candidate in anchor_candidates:
                            anchor_mask |= np.asarray(candidate.mask, dtype=bool)
                        anchor_components, residual_mask, split_flags, shape_diagnostics = (
                            split_anchor_residual(
                                whole.mask,
                                anchor_mask,
                                max_anchor_instances=max_instances,
                                minimum_pixels=minimum_pixels,
                            )
                        )
                    semantic, instance, components, combine_flags = (
                        combine_category_candidates(
                            anchor_components,
                            residual_mask,
                            (image.height, image.width),
                            anchor_part=str(category_config["anchor_part"]),
                            residual_part=str(category_config["residual_part"]),
                            residual_label=str(category_config["residual_label"]),
                            score=whole.score,
                            box_xyxy=whole.box_xyxy,
                            minimum_pixels=minimum_pixels,
                        )
                    )
                    flags = sorted(set(split_flags + combine_flags))
            if int(np.count_nonzero(semantic)) > image.width * image.height * 0.7:
                flags = sorted(set(flags + ["마스크_면적_과다_검수필요"]))

            safe_id = source_id.replace("/", "__")
            semantic_path = staging / "semantic_candidates" / f"{safe_id}.png"
            instance_path = staging / "instance_candidates" / f"{safe_id}.png"
            overlay_path = staging / "overlays" / f"{safe_id}.jpg"
            save_mask(semantic_path, semantic)
            save_mask(instance_path, instance)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            _render_overlay(image, instance, components).save(overlay_path, quality=92)
            rows.append(
                {
                    "source_dataset": "custom",
                    "source_id": source_id,
                    "object_id": record["object_id"],
                    "split": record["split"],
                    "image_path": record["image_path"],
                    "candidate_semantic_path": semantic_path.relative_to(staging).as_posix(),
                    "candidate_instance_path": instance_path.relative_to(staging).as_posix(),
                    "overlay_path": overlay_path.relative_to(staging).as_posix(),
                    "category": record.get("category"),
                    "components": components,
                    "shape_split": shape_diagnostics,
                    "review_flags": flags,
                    "annotation_status": "human_review_required",
                    "conversion_status": "pseudolabel_candidate_only",
                    "mapped_labels": sorted({str(c["label"]) for c in components}),
                    "functional_region_generated": any(
                        str(c["label"]) == "functional_region" for c in components
                    ),
                }
            )
            print(f"[{number}/{len(records)}] 후보 생성: {source_id} / 검수표시={flags}")

        _write_jsonl(staging / "candidate_manifest.jsonl", rows)
        summary = {
            "status": "human_review_required",
            "processed_images": len(rows),
            "images_with_candidates": sum(bool(row["components"]) for row in rows),
            "images_missing_handle": sum(
                not any(
                    component["part"] == "handle" for component in row["components"]
                )
                for row in rows
            ),
            "images_missing_body": sum(
                not any(component["part"] == "body" for component in row["components"])
                for row in rows
                if str(row.get("category")) in ("mug", "insulated_mug", "None")
            ),
            "images_missing_functional": sum(
                not row["functional_region_generated"]
                for row in rows
                if str(row.get("category")) in ("scissors", "screwdriver")
            ),
            "images_with_functional": sum(
                bool(row["functional_region_generated"]) for row in rows
            ),
            "images_with_review_flags": sum(bool(row["review_flags"]) for row in rows),
            "models": {"grounding_dino": detector_id, "sam2": sam2_id},
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": args.device,
            "thresholds": thresholds,
            "policy": config["policy"],
            "warning": "이 결과는 정답이 아니며 사람이 경계를 검수하기 전에는 학습에 사용하지 않습니다.",
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        priority_rows = [row for row in rows if row["review_flags"]]
        priority_root = staging / "priority_overlays"
        priority_root.mkdir(parents=True, exist_ok=True)
        review_lines = [
            "# 자동 표시된 우선 검수 목록",
            "# 이 목록에 없더라도 모든 반자동 후보는 사람 검수가 필요합니다.",
            "",
        ]
        for row in priority_rows:
            source_overlay = staging / str(row["overlay_path"])
            target_overlay = priority_root / source_overlay.name
            shutil.copy2(source_overlay, target_overlay)
            review_lines.append(
                f"{row['source_id']} | {', '.join(row['review_flags'])}"
            )
        (staging / "검수_우선목록.txt").write_text(
            "\n".join(review_lines) + "\n", encoding="utf-8"
        )
        staging.replace(output_root)
    except BaseException:
        # 실패 원인을 확인할 수 있도록 중간 결과는 지우지 않고 임시 폴더에 보존한다.
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"반자동 후보를 저장했습니다: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
