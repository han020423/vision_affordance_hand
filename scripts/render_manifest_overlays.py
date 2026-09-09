#!/usr/bin/env python
"""manifest에서 클래스 균형을 맞춘 결정적 오버레이 샘플을 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.render_mask_overlay import render_overlay  # noqa: E402


CLASSES = ("grasp_region", "functional_region")


def evenly_spaced(records: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """source ID로 정렬된 후보 전체 구간에서 결정적으로 샘플을 선택한다."""

    if count <= 0:
        raise ValueError("count는 양수여야 합니다")
    ordered = sorted(records, key=lambda record: str(record["source_id"]))
    if len(ordered) < count:
        raise ValueError(f"후보 {count}개가 필요하지만 {len(ordered)}개만 찾았습니다")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def render_contact_sheet(paths: list[Path], output_path: Path) -> None:
    """생성한 오버레이로 GUI 없이 확인 가능한 간결한 품질 점검표를 만든다."""

    columns = 5
    tile_width, tile_height = 240, 210
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((tile_width, tile_height - 30))
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            sheet.paste(preview, (x, y))
            draw.text((x + 4, y + tile_height - 26), path.stem[:36], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    converted = [record for record in records if record["conversion_status"] == "converted"]

    audit_rows: list[dict[str, str]] = []
    for class_name in CLASSES:
        candidates = [
            record for record in converted if class_name in record.get("mapped_labels", [])
        ]
        selected = evenly_spaced(candidates, args.per_class)
        overlay_paths: list[Path] = []
        for record in selected:
            source_id = str(record["source_id"])
            safe_id = source_id.replace("/", "__")
            output_path = args.output / class_name / f"{safe_id}.png"
            image_path = REPOSITORY_ROOT / str(record["image_path"])
            mask_path = REPOSITORY_ROOT / str(record["semantic_mask_path"])
            render_overlay(image_path, mask_path, output_path, args.alpha)
            overlay_paths.append(output_path)
            audit_rows.append(
                {
                    "audit_class": class_name,
                    "source_id": source_id,
                    "image_path": str(record["image_path"]),
                    "semantic_mask_path": str(record["semantic_mask_path"]),
                    "overlay_path": output_path.as_posix(),
                }
            )
        render_contact_sheet(overlay_paths, args.output / f"contact_sheet_{class_name}.jpg")
        print(f"{class_name}: 오버레이 {len(overlay_paths)}장")

    audit_path = args.output / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(audit_path)
    print(f"감사 manifest: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
