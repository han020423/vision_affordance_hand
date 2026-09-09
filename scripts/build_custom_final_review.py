"""수정본과 정상 후보를 합쳐 자체 머그 140장 최종 확인 세트를 만든다."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_review import (  # noqa: E402
    inspect_mug_instance_mask,
    load_candidate_manifest,
    remove_tiny_instance_fragments,
    render_review_overlay,
    semantic_from_instances,
    standardize_mug_instances,
)


def parse_args() -> argparse.Namespace:
    """입력 후보·수정본·출력 경로를 설정한다."""

    parser = argparse.ArgumentParser(
        description="자체 머그 수정본과 정상 후보를 합쳐 최종 육안 검수 세트를 만듭니다."
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("outputs/custom_pseudolabel_review/full_shape_split_v2"),
        help="전체 140장 반자동 후보 폴더",
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=Path("outputs/custom_mask_review/review_20260818"),
        help="사람이 수정한 64장 작업 폴더",
    )
    parser.add_argument(
        "--raw-image-root",
        type=Path,
        default=Path("data/raw/custom/images"),
        help="자체 촬영 원본 이미지 폴더",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/custom_final_review/final_140_v1"),
        help="최종 전체 확인 세트를 저장할 새 폴더",
    )
    parser.add_argument(
        "--minimum-fragment-pixels",
        type=int,
        default=64,
        help="브러시 잔여점으로 제거할 연결 성분 최소 크기",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """상대경로를 저장소 루트 기준 절대경로로 바꾼다."""

    return path if path.is_absolute() else REPOSITORY_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """빈 행을 건너뛰며 JSONL을 읽는다."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """한글 검수 상태를 보존해 JSONL을 저장한다."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_image_required(path: Path, image: object) -> None:
    """OpenCV가 저장 실패를 반환하면 경로와 함께 중단한다."""

    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")


def find_raw_images(raw_root: Path) -> dict[str, Path]:
    """원본 파일명을 색인하고 중복 파일명은 거부한다."""

    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    result: dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if path.name in result:
            raise ValueError(f"원본 이미지 파일명이 중복됩니다: {path.name}")
        result[path.name] = path
    return result


def create_contact_sheets(
    overlay_paths: list[Path],
    output_root: Path,
    *,
    columns: int = 4,
    rows_per_sheet: int = 4,
) -> int:
    """폴더를 하나씩 열지 않아도 되도록 전체 오버레이 모음 이미지를 만든다."""

    output_root.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, label_height = 420, 236, 28
    page_size = columns * rows_per_sheet
    pages = 0
    for page_start in range(0, len(overlay_paths), page_size):
        page_paths = overlay_paths[page_start : page_start + page_size]
        canvas = Image.new(
            "RGB",
            (columns * tile_width, rows_per_sheet * (tile_height + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for local_index, overlay_path in enumerate(page_paths):
            row_index, column_index = divmod(local_index, columns)
            with Image.open(overlay_path).convert("RGB") as image:
                fitted = ImageOps.contain(image, (tile_width, tile_height))
                x = column_index * tile_width + (tile_width - fitted.width) // 2
                y = row_index * (tile_height + label_height)
                canvas.paste(fitted, (x, y))
            label = overlay_path.stem
            draw.text(
                (column_index * tile_width + 5, y + tile_height + 5),
                label,
                fill="black",
            )
        pages += 1
        canvas.save(output_root / f"전체검수_{pages:02d}.jpg", quality=92)
    return pages


def main() -> int:
    """입력을 검증하고 원본을 보존한 최종 확인용 결과를 생성한다."""

    args = parse_args()
    candidate_root = resolve_path(args.candidate_root)
    correction_root = resolve_path(args.correction_root)
    raw_root = resolve_path(args.raw_image_root)
    output_root = resolve_path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"기존 최종 검수 결과를 덮어쓰지 않습니다: {output_root}")

    candidates = load_candidate_manifest(candidate_root / "candidate_manifest.jsonl")
    corrections = read_jsonl(correction_root / "review_manifest.jsonl")
    incomplete = [
        str(row["source_id"])
        for row in corrections
        if row.get("review_status") not in {"human_corrected", "excluded_ambiguous"}
    ]
    if incomplete:
        raise ValueError("수정이 끝나지 않은 항목: " + ", ".join(incomplete))
    correction_by_name = {
        Path(str(row["image_path"])).name: row for row in corrections
    }
    raw_by_name = find_raw_images(raw_root)

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 결과를 먼저 확인하세요: {staging}")
    for directory in (
        "images",
        "instance_masks",
        "semantic_masks",
        "overlays",
        "priority_overlays",
        "contact_sheets",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    priority_lines = [
        "# 자동 구조 검사에서 다시 확인이 필요한 최종 오버레이",
        "# 손잡이 비가시는 오류가 아니라 카메라 반대편에 가려진 정상 장면일 수 있습니다.",
        "",
    ]
    excluded = 0
    for candidate in candidates:
        filename = Path(str(candidate["overlay_path"])).name
        source_name = filename.split("__", maxsplit=1)[-1]
        raw_path = raw_by_name.get(source_name)
        if raw_path is None:
            raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {source_name}")

        correction = correction_by_name.get(filename)
        if correction and correction["review_status"] == "excluded_ambiguous":
            excluded += 1
            rows.append(
                {
                    "source_dataset": "custom",
                    "source_id": candidate["source_id"],
                    "object_id": candidate["object_id"],
                    "split": candidate["split"],
                    "annotation_status": "excluded_ambiguous",
                    "conversion_status": "excluded_after_human_review",
                    "exclusion_reason": correction["review_note"],
                }
            )
            continue

        if correction:
            mask_path = correction_root / str(correction["corrected_instance_path"])
            instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            source_kind = "human_corrected"
            review_note = correction["review_note"]
        else:
            mask_path = candidate_root / str(candidate["candidate_instance_path"])
            raw_candidate_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw_candidate_mask is None:
                raise FileNotFoundError(f"후보 마스크를 읽을 수 없습니다: {mask_path}")
            instance_mask = standardize_mug_instances(
                raw_candidate_mask, candidate.get("components", [])
            )
            source_kind = "visually_accepted_candidate"
            review_note = None
        if instance_mask is None:
            raise FileNotFoundError(f"수정 마스크를 읽을 수 없습니다: {mask_path}")

        cleaned_mask, removed = remove_tiny_instance_fragments(
            instance_mask, minimum_pixels=args.minimum_fragment_pixels
        )
        components, validation_flags = inspect_mug_instance_mask(
            cleaned_mask, minimum_pixels=args.minimum_fragment_pixels
        )
        image_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"원본 이미지를 읽을 수 없습니다: {raw_path}")
        if image_bgr.shape[:2] != cleaned_mask.shape:
            raise ValueError(f"이미지와 마스크 크기가 다릅니다: {candidate['source_id']}")

        image_target = staging / "images" / filename
        instance_target = staging / "instance_masks" / Path(filename).with_suffix(".png")
        semantic_target = staging / "semantic_masks" / instance_target.name
        overlay_target = staging / "overlays" / filename
        shutil.copy2(raw_path, image_target)
        write_image_required(instance_target, cleaned_mask)
        write_image_required(semantic_target, semantic_from_instances(cleaned_mask))
        write_image_required(
            overlay_target, render_review_overlay(image_bgr, cleaned_mask)
        )
        overlay_paths.append(overlay_target)

        if validation_flags:
            shutil.copy2(overlay_target, staging / "priority_overlays" / filename)
            priority_lines.append(
                f"{candidate['source_id']} | {', '.join(validation_flags)}"
            )
        rows.append(
            {
                "source_dataset": "custom",
                "source_id": candidate["source_id"],
                "object_id": candidate["object_id"],
                "split": candidate["split"],
                "image_path": image_target.relative_to(staging).as_posix(),
                "semantic_mask_path": semantic_target.relative_to(staging).as_posix(),
                "instance_mask_path": instance_target.relative_to(staging).as_posix(),
                "overlay_path": overlay_target.relative_to(staging).as_posix(),
                "components": components,
                "validation_flags": validation_flags,
                "tiny_fragment_pixels_removed": {
                    "handle": removed.get(1, 0),
                    "body": removed.get(2, 0),
                },
                "label_source": source_kind,
                "review_note": review_note,
                "annotation_status": "final_confirmation_required",
                "conversion_status": "final_review_candidate_only",
                "mapped_labels": ["grasp_region"],
                "functional_region_generated": False,
            }
        )

    write_jsonl(staging / "final_review_manifest.jsonl", rows)
    (staging / "우선_재확인목록.txt").write_text(
        "\n".join(priority_lines) + "\n", encoding="utf-8"
    )
    pages = create_contact_sheets(overlay_paths, staging / "contact_sheets")
    included_rows = [row for row in rows if row.get("instance_mask_path")]
    summary = {
        "status": "final_confirmation_required",
        "candidate_images": len(candidates),
        "included_images": len(included_rows),
        "excluded_images": excluded,
        "human_corrected_images": sum(
            row.get("label_source") == "human_corrected" for row in included_rows
        ),
        "visually_accepted_candidate_images": sum(
            row.get("label_source") == "visually_accepted_candidate"
            for row in included_rows
        ),
        "images_without_visible_handle": sum(
            "손잡이_비가시_또는_라벨없음" in row.get("validation_flags", [])
            for row in included_rows
        ),
        "images_with_validation_flags": sum(
            bool(row.get("validation_flags")) for row in included_rows
        ),
        "by_object": dict(Counter(str(row["object_id"]) for row in included_rows)),
        "by_split": dict(Counter(str(row["split"]) for row in included_rows)),
        "contact_sheet_pages": pages,
        "classes": {"0": "grasp_region", "1": "functional_region"},
        "functional_region_generated": False,
        "warning": "전체 최종 오버레이를 확인하기 전에는 학습 데이터로 승격하지 않습니다.",
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
