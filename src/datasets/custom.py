"""자체 촬영 이미지를 물체 ID 기준 라벨링 작업공간으로 준비한다."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from src.datasets.common import SampleRecord, relative_or_absolute, save_mask, write_jsonl


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASSES = {0: "grasp_region", 1: "functional_region"}
FILENAME_PATTERN = re.compile(
    r"^WIN_(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_"
    r"(?P<second>\d{2})_Pro(?: \(\d+\))?$"
)


@dataclass(frozen=True)
class CaptureAssignment:
    """한 촬영 구간에 고정된 물체 ID, 장면과 데이터 분할."""

    capture_id: str
    object_id: str
    category: str
    split: str
    scene: str
    start: datetime
    end: datetime
    expected_count: int
    is_negative: bool = False

    def contains(self, timestamp: datetime) -> bool:
        """파일 촬영시각이 양 끝을 포함한 구간 안에 있는지 반환한다."""

        return self.start <= timestamp <= self.end


@dataclass
class CustomSampleRecord(SampleRecord):
    """자체 데이터 재현성과 라벨링 상태를 추가로 기록한다."""

    category: str = ""
    capture_session: str = ""
    scene: str = ""
    image_width: int | None = None
    image_height: int | None = None
    image_sha256: str = ""
    labeling_image_path: str = ""
    annotation_status: str = "pending"
    is_negative: bool = False


def parse_capture_timestamp(path: Path) -> datetime | None:
    """Windows 카메라 파일명에서 촬영시각을 읽고 다른 형식은 추측하지 않는다."""

    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None
    value = (
        match.group("date")
        + match.group("hour")
        + match.group("minute")
        + match.group("second")
    )
    return datetime.strptime(value, "%Y%m%d%H%M%S")


def _repository_path(repository_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_custom_capture_config(
    config_path: Path,
    repository_root: Path,
) -> tuple[dict[str, object], list[CaptureAssignment]]:
    """자체 촬영 설정을 읽고 라벨 정책 및 구간 중첩을 엄격히 검증한다."""

    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    classes = {int(key): str(value) for key, value in (document.get("classes") or {}).items()}
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"자체 데이터 클래스는 {EXPECTED_CLASSES}이어야 합니다. 현재 값: {classes}"
        )
    if int(document.get("ignore_index", -1)) != 255:
        raise ValueError("자체 데이터 ignore_index는 255여야 합니다")

    assignments: list[CaptureAssignment] = []
    seen_capture_ids: set[str] = set()
    for row in document.get("captures", []):
        capture_id = str(row["capture_id"])
        if capture_id in seen_capture_ids:
            raise ValueError(f"capture_id가 중복되었습니다: {capture_id}")
        seen_capture_ids.add(capture_id)
        assignment = CaptureAssignment(
            capture_id=capture_id,
            object_id=str(row["object_id"]),
            category=str(row["category"]),
            split=str(row["split"]),
            scene=str(row["scene"]),
            start=datetime.fromisoformat(str(row["start"])),
            end=datetime.fromisoformat(str(row["end"])),
            expected_count=int(row["expected_count"]),
            is_negative=bool(row.get("is_negative", False)),
        )
        if assignment.start > assignment.end:
            raise ValueError(f"촬영 시작이 종료보다 늦습니다: {capture_id}")
        if assignment.expected_count < 1:
            raise ValueError(f"expected_count는 1 이상이어야 합니다: {capture_id}")
        if assignment.split not in {"train", "validation", "test"}:
            raise ValueError(f"지원하지 않는 split입니다: {capture_id}={assignment.split}")
        assignments.append(assignment)

    for index, left in enumerate(assignments):
        for right in assignments[index + 1 :]:
            if max(left.start, right.start) <= min(left.end, right.end):
                raise ValueError(
                    f"촬영 구간이 중첩되었습니다: {left.capture_id}, {right.capture_id}"
                )
    if not assignments:
        raise ValueError(f"captures가 비어 있습니다: {config_path}")
    return document, assignments


def _assignment_for(
    timestamp: datetime | None,
    assignments: list[CaptureAssignment],
) -> CaptureAssignment | None:
    if timestamp is None:
        return None
    matches = [assignment for assignment in assignments if assignment.contains(timestamp)]
    if len(matches) > 1:
        raise RuntimeError(f"한 파일이 여러 촬영 구간에 포함되었습니다: {timestamp.isoformat()}")
    return matches[0] if matches else None


def _labeling_guide(reserved_test_object_id: str) -> str:
    """작업공간에 함께 둘 고정 라벨 정책 안내서를 만든다."""

    return f"""# 자체 촬영 데이터 라벨링 안내

## 고정 클래스

- `0: grasp_region`
- `1: functional_region`
- `ignore: 255`

## 머그·컵 라벨 규칙

1. 손잡이와 잡을 수 있는 몸통은 각각 별도의 `grasp_region` 컴포넌트로 유지한다.
2. 손잡이와 몸통을 하나의 마스크로 합치지 않는다.
3. 컵 몸통을 `functional_region`으로 지정하지 않는다.
4. 컵 입구·내부·뚜껑처럼 의미가 불명확한 부분은 억지로 지정하지 않고 `ignore` 또는 사람 검수로 남긴다.
5. 배경 사진은 객체 폴리곤을 만들지 않는다.

## 저장값

- semantic PNG: 배경 0, grasp 1, functional 2, ignore 255
- instance PNG: 배경 0, 연결 성분마다 1 이상의 서로 다른 정수 ID

객체 사진은 아직 정답 마스크가 없으므로 `pending_annotation` 상태다. 자동 예측을 정답처럼
사용하지 말고 사람이 최종 경계를 확인해야 한다. 최종 Test용 예약 물체 ID는
`{reserved_test_object_id}`이며, 해당 실제 물체를 촬영하기 전에는 Test 결과를 만들지 않는다.
"""


def prepare_custom_dataset(
    config_path: Path,
    repository_root: Path,
    *,
    dry_run: bool = False,
) -> tuple[list[CustomSampleRecord], dict[str, object]]:
    """원본을 보존하며 자체 이미지 manifest와 라벨링 준비본을 결정적으로 만든다."""

    document, assignments = load_custom_capture_config(config_path, repository_root)
    raw_root = _repository_path(repository_root, document["raw_image_root"])
    output_root = _repository_path(repository_root, document["labeling_output_root"])
    manifest_path = _repository_path(repository_root, document["manifest_path"])
    reserved_test_object_id = str(document.get("reserved_test_object_id", "mug_04"))
    strict_unassigned = bool(document.get("strict_unassigned", True))

    if not raw_root.is_dir():
        raise FileNotFoundError(f"자체 촬영 이미지 폴더가 없습니다: {raw_root}")

    image_paths = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"자체 촬영 이미지를 찾지 못했습니다: {raw_root}")

    assignment_counts: Counter[str] = Counter()
    records: list[CustomSampleRecord] = []
    source_paths: dict[str, Path] = {}
    file_hashes: dict[str, list[str]] = {}

    for image_path in image_paths:
        timestamp = parse_capture_timestamp(image_path)
        assignment = _assignment_for(timestamp, assignments)
        if assignment is None:
            if strict_unassigned:
                raise ValueError(
                    "설정된 촬영 구간에 포함되지 않는 파일입니다. "
                    f"추측해서 배정하지 않습니다: {image_path.name}"
                )
            continue

        assignment_counts[assignment.capture_id] += 1
        safe_name = image_path.name
        labeling_relative = Path("images") / assignment.object_id / safe_name
        semantic_relative = (
            Path("semantic_masks") / assignment.object_id / f"{image_path.stem}.png"
        )
        instance_relative = (
            Path("instance_masks") / assignment.object_id / f"{image_path.stem}.png"
        )
        image_hash = _sha256_file(image_path)
        file_hashes.setdefault(image_hash, []).append(image_path.name)

        width: int | None = None
        height: int | None = None
        read_error: str | None = None
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            read_error = type(exc).__name__

        if read_error is not None:
            conversion_status = "excluded"
            annotation_status = "unreadable"
            reasons = [f"unreadable_image:{read_error}"]
            mapped_labels: list[str] = []
            semantic_path: str | None = None
            instance_path: str | None = None
        elif assignment.is_negative:
            conversion_status = "converted"
            annotation_status = "verified_empty"
            reasons = ["verified_background_negative"]
            mapped_labels = ["background"]
            semantic_path = relative_or_absolute(output_root / semantic_relative, repository_root)
            instance_path = relative_or_absolute(output_root / instance_relative, repository_root)
        else:
            conversion_status = "pending_annotation"
            annotation_status = "human_annotation_required"
            reasons = ["human_affordance_annotation_required"]
            mapped_labels = []
            semantic_path = relative_or_absolute(output_root / semantic_relative, repository_root)
            instance_path = relative_or_absolute(output_root / instance_relative, repository_root)

        source_id = f"{assignment.object_id}/{image_path.stem}"
        record = CustomSampleRecord(
            source_dataset="custom",
            source_id=source_id,
            object_id=assignment.object_id,
            object_id_status=(
                "no_physical_object_negative"
                if assignment.is_negative
                else "verified_physical_object"
            ),
            image_path=relative_or_absolute(image_path, repository_root),
            source_mask_path="",
            semantic_mask_path=semantic_path,
            instance_mask_path=instance_path,
            original_labels=[] if assignment.is_negative else ["unlabeled_custom_capture"],
            mapped_labels=mapped_labels,
            split=assignment.split,
            conversion_status=conversion_status,
            selection_status="selected",
            reasons=reasons,
            components=[],
            category=assignment.category,
            capture_session=assignment.capture_id,
            scene=assignment.scene,
            image_width=width,
            image_height=height,
            image_sha256=image_hash,
            labeling_image_path=relative_or_absolute(
                output_root / labeling_relative, repository_root
            ),
            annotation_status=annotation_status,
            is_negative=assignment.is_negative,
        )
        records.append(record)
        source_paths[source_id] = image_path

    for assignment in assignments:
        actual = assignment_counts[assignment.capture_id]
        if actual != assignment.expected_count:
            raise ValueError(
                f"촬영 구간 {assignment.capture_id}의 파일 수가 다릅니다: "
                f"예상={assignment.expected_count}, 실제={actual}"
            )

    duplicate_groups = [names for names in file_hashes.values() if len(names) > 1]
    if duplicate_groups:
        raise ValueError(f"완전히 동일한 자체 이미지가 있습니다: {duplicate_groups}")

    status_counts = Counter(record.conversion_status for record in records)
    object_counts = Counter(record.object_id for record in records)
    split_counts = Counter(record.split for record in records)
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_dataset": "custom",
        "source_image_count": len(records),
        "object_counts": dict(sorted(object_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "capture_counts": dict(sorted(assignment_counts.items())),
        "reserved_test_object_id": reserved_test_object_id,
        "final_test_status": "pending_physical_object_capture",
        "policy": {
            "classes": EXPECTED_CLASSES,
            "ignore_index": 255,
            "mug_body_is_not_functional_region": True,
            "separate_grasp_components": True,
        },
    }

    if dry_run:
        return records, summary
    if output_root.exists():
        raise FileExistsError(
            f"라벨링 출력 경로가 이미 있습니다. 기존 결과를 보존했습니다: {output_root}"
        )

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 출력 경로가 남아 있습니다: {staging}")
    try:
        for record in records:
            if record.conversion_status == "excluded":
                continue
            source_path = source_paths[record.source_id]
            target_image = staging / Path(record.labeling_image_path).relative_to(
                Path(relative_or_absolute(output_root, repository_root))
            )
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_image)

            if record.is_negative:
                assert record.image_width is not None and record.image_height is not None
                shape = (record.image_height, record.image_width)
                semantic_target = staging / Path(str(record.semantic_mask_path)).relative_to(
                    Path(relative_or_absolute(output_root, repository_root))
                )
                instance_target = staging / Path(str(record.instance_mask_path)).relative_to(
                    Path(relative_or_absolute(output_root, repository_root))
                )
                save_mask(semantic_target, np.zeros(shape, dtype=np.uint8))
                save_mask(instance_target, np.zeros(shape, dtype=np.uint16))

        (staging / "라벨링_안내.md").write_text(
            _labeling_guide(reserved_test_object_id), encoding="utf-8"
        )
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
        write_jsonl(manifest_path, records)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return records, summary

