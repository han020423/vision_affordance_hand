#!/usr/bin/env python
"""UMD 보류 용기 이미지를 별도의 contain 정책 감사용 묶음으로 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS = ("cup_03", "mug_20", "pot_02")


def resolve_repository_path(value: str) -> Path:
    """manifest의 경로를 저장소 루트를 기준으로 해석한다."""

    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/umd.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 출력 디렉터리를 재사용하지 않습니다: {output_dir}")
    selected_objects = set(args.objects)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "semantic_masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # 공식 test 지표와 섞이지 않도록 정책 확인에 필요한 물체만 별도 묶음으로 복사한다.
    output_rows: list[dict[str, object]] = []
    source_counts = {object_id: 0 for object_id in sorted(selected_objects)}
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        object_id = str(row.get("object_id", ""))
        if object_id not in selected_objects:
            continue
        if row.get("split") != "test":
            raise ValueError(f"선택한 물체가 test 분할이 아닙니다({line_number}번째 줄): {object_id}")
        if row.get("conversion_status") != "converted":
            continue
        if "umd_contain_policy" not in row.get("reasons", []):
            raise ValueError(f"선택한 행에 contain 정책 사유가 없습니다: {line_number}번째 줄")
        image_source = resolve_repository_path(str(row["image_path"]))
        mask_source = resolve_repository_path(str(row["semantic_mask_path"]))
        if not image_source.is_file() or not mask_source.is_file():
            raise FileNotFoundError(f"감사 원본 이미지·마스크 쌍이 없습니다: {line_number}번째 줄")
        semantic = np.asarray(Image.open(mask_source).convert("L"))
        if not np.any(semantic == 255):
            raise ValueError(f"contain 정책 샘플에 ignore 픽셀이 없습니다: {row['source_id']}")

        safe_name = str(row["source_id"]).replace("/", "__")
        image_destination = images_dir / f"{safe_name}{image_source.suffix.lower()}"
        mask_destination = masks_dir / f"{safe_name}.png"
        shutil.copy2(image_source, image_destination)
        shutil.copy2(mask_source, mask_destination)
        output_rows.append(
            {
                "source_dataset": "umd",
                "source_id": row["source_id"],
                "object_id": object_id,
                "split": "test",
                "conversion_status": row["conversion_status"],
                "original_labels": row["original_labels"],
                "mapped_labels": row["mapped_labels"],
                "reasons": row["reasons"],
                "image_path": image_destination.relative_to(output_dir).as_posix(),
                "semantic_mask_path": mask_destination.relative_to(output_dir).as_posix(),
            }
        )
        source_counts[object_id] += 1

    missing = sorted(object_id for object_id, count in source_counts.items() if count == 0)
    if missing:
        raise ValueError(f"정책 감사에 사용할 수 있는 샘플이 없습니다: {missing}")
    output_rows.sort(key=lambda row: str(row["source_id"]))
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "purpose": "공식 YOLO test 지표에 포함하지 않는 별도 UMD contain 정책 감사",
                "policy": "contain -> ignore",
                "objects": source_counts,
                "sample_count": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "objects": source_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
