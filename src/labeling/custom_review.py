"""자체 머그 후보 마스크의 사람 검수 작업 공간을 준비하는 도우미."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


HANDLE_INSTANCE_ID = 1
BODY_INSTANCE_ID = 2
# 2026-08-21 확장: 가위 날·드라이버 축 같은 기능 부위 인스턴스.
FUNCTIONAL_INSTANCE_ID = 3
GRASP_SEMANTIC_VALUE = 1
FUNCTIONAL_SEMANTIC_VALUE = 2


def load_review_notes(config_path: Path) -> tuple[dict[str, str], dict[str, object]]:
    """사용자가 기록한 파일별 검수 의견을 읽고 기본 형식을 검사한다."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("reviews"), dict):
        raise ValueError("검수 설정에는 reviews 매핑이 필요합니다")

    reviews: dict[str, str] = {}
    for filename, note in config["reviews"].items():
        name = str(filename).strip()
        description = str(note).strip()
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError(f"검수 파일명의 확장자를 확인하세요: {name}")
        if not description:
            raise ValueError(f"검수 사유가 비어 있습니다: {name}")
        reviews[name] = description
    return reviews, config


def load_candidate_manifest(manifest_path: Path) -> list[dict[str, object]]:
    """반자동 후보 JSONL을 순서를 보존해 읽는다."""

    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"후보 manifest {line_number}행을 읽을 수 없습니다: {error}"
                ) from error
            rows.append(row)
    return rows


def standardize_mug_instances(
    candidate_mask: np.ndarray,
    components: Iterable[dict[str, object]],
) -> np.ndarray:
    """후보마다 달라질 수 있는 ID를 손잡이=1, 몸통=2로 고정한다.

    최종 의미 클래스는 둘 다 ``grasp_region``이다. 여기서 ID를 나누는 목적은
    파지 후보 두 개를 수동 수정 화면에서도 서로 독립적으로 유지하기 위해서다.
    """

    if candidate_mask.ndim != 2:
        raise ValueError("인스턴스 마스크는 단일 채널이어야 합니다")
    standardized = np.zeros(candidate_mask.shape, dtype=np.uint8)
    part_to_id = {"handle": HANDLE_INSTANCE_ID, "body": BODY_INSTANCE_ID}
    for component in components:
        part = str(component.get("part", ""))
        if part not in part_to_id:
            raise ValueError(f"알 수 없는 머그 부위입니다: {part}")
        source_id = int(component["instance_id"])
        standardized[candidate_mask == source_id] = part_to_id[part]
    return standardized


def semantic_from_instances(instance_mask: np.ndarray) -> np.ndarray:
    """인스턴스 마스크를 저장용 semantic 값으로 변환한다.

    손잡이(1)·몸통(2)은 grasp 값 1, 기능 부위(3)는 functional 값 2가 된다.
    머그 전용이던 이전 동작(모두 1)과 손잡이·몸통에 대해서는 완전히 같다.
    """

    semantic = np.zeros(instance_mask.shape, dtype=np.uint8)
    semantic[
        (instance_mask == HANDLE_INSTANCE_ID) | (instance_mask == BODY_INSTANCE_ID)
    ] = GRASP_SEMANTIC_VALUE
    semantic[instance_mask == FUNCTIONAL_INSTANCE_ID] = FUNCTIONAL_SEMANTIC_VALUE
    return semantic


def standardize_category_instances(
    candidate_mask: np.ndarray,
    components: Iterable[dict[str, object]],
) -> np.ndarray:
    """카테고리 후보의 ID를 손잡이=1, 몸통=2, 기능=3으로 고정한다.

    가위 고리처럼 손잡이 컴포넌트가 여러 개면 모두 값 1을 공유한다.
    서로 떨어진 연결 성분이므로 이후 변환 단계에서 컴포넌트 단위로
    다시 분리되며, 이는 승인 manifest에 기록된 기존 관례와 같다.
    """

    if candidate_mask.ndim != 2:
        raise ValueError("인스턴스 마스크는 단일 채널이어야 합니다")
    standardized = np.zeros(candidate_mask.shape, dtype=np.uint8)
    for component in components:
        label = str(component.get("label", ""))
        part = str(component.get("part", ""))
        source_id = int(component["instance_id"])
        if label == "functional_region":
            target = FUNCTIONAL_INSTANCE_ID
        elif label == "grasp_region" and part == "body":
            target = BODY_INSTANCE_ID
        elif label == "grasp_region":
            target = HANDLE_INSTANCE_ID
        else:
            raise ValueError(f"알 수 없는 컴포넌트 라벨입니다: {label}/{part}")
        standardized[candidate_mask == source_id] = target
    return standardized


# 카테고리별 저장 규칙: 필수/금지 부위. 편집기 저장 검증과 테스트가 함께 쓴다.
REVIEW_RULES_BY_CATEGORY: dict[str, dict[str, bool]] = {
    "mug": {"body_required": True, "functional_required": False, "functional_forbidden": True},
    "insulated_mug": {"body_required": True, "functional_required": False, "functional_forbidden": True},
    "scissors": {"body_required": False, "functional_required": True, "functional_forbidden": False},
    "screwdriver": {"body_required": False, "functional_required": True, "functional_forbidden": False},
}


def validate_review_masks(
    category: str,
    *,
    handle_pixels: int,
    body_pixels: int,
    functional_pixels: int,
    handle_not_visible: bool,
) -> list[str]:
    """저장 전 카테고리 규칙 위반을 사람이 읽을 메시지 목록으로 반환한다.

    빈 목록이면 저장해도 된다. 알 수 없는 카테고리는 보수적으로 머그 규칙을
    쓰지 않고 명시적으로 거부해 정책 없는 라벨 생성을 막는다.
    """

    rules = REVIEW_RULES_BY_CATEGORY.get(category)
    if rules is None:
        return [f"저장 규칙이 정의되지 않은 카테고리입니다: {category}"]
    problems: list[str] = []
    if rules["body_required"] and body_pixels == 0:
        problems.append("몸통 영역은 반드시 필요합니다.")
    if not rules["body_required"] and body_pixels > 0:
        problems.append("이 카테고리에는 몸통(B) 부위를 쓰지 않습니다. E로 지우거나 H/F로 바꾸세요.")
    if rules["functional_required"] and functional_pixels == 0:
        problems.append("기능 부위(F, 날/축)가 반드시 필요합니다.")
    if rules["functional_forbidden"] and functional_pixels > 0:
        problems.append("머그에는 기능 부위(F)를 만들지 않습니다. E로 지우세요.")
    if handle_pixels == 0 and not handle_not_visible:
        problems.append("손잡이가 실제로 보이지 않는 장면이면 V를 누른 뒤 다시 저장하세요.")
    if handle_pixels > 0 and handle_not_visible:
        problems.append("손잡이 픽셀이 남아 있습니다. E로 지우거나 V를 해제하세요.")
    return problems


def remove_tiny_instance_fragments(
    instance_mask: np.ndarray,
    *,
    minimum_pixels: int = 64,
) -> tuple[np.ndarray, dict[int, int]]:
    """각 인스턴스에서 최소 크기보다 작은 브러시 잔여점만 제거한다.

    큰 연결 성분을 임의로 합치거나 삭제하지 않는다. 손잡이가 가려져 여러 조각으로
    보일 수 있기 때문이다. 제거한 픽셀 수는 최종 manifest에 감사 정보로 남긴다.
    """

    if minimum_pixels < 1:
        raise ValueError("minimum_pixels는 1 이상이어야 합니다")
    cleaned = np.asarray(instance_mask, dtype=np.uint8).copy()
    removed: dict[int, int] = {}
    for instance_id in (HANDLE_INSTANCE_ID, BODY_INSTANCE_ID, FUNCTIONAL_INSTANCE_ID):
        binary = (cleaned == instance_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        removed_pixels = 0
        for local_id in range(1, count):
            area = int(stats[local_id, cv2.CC_STAT_AREA])
            if area < minimum_pixels:
                removed_pixels += area
                cleaned[labels == local_id] = 0
        removed[instance_id] = removed_pixels
    return cleaned, removed


def inspect_mug_instance_mask(
    instance_mask: np.ndarray,
    *,
    minimum_pixels: int = 64,
) -> tuple[list[dict[str, object]], list[str]]:
    """최종 확인 전에 머그 인스턴스의 구조적 이상을 검사한다."""

    allowed = {0, HANDLE_INSTANCE_ID, BODY_INSTANCE_ID}
    values = set(np.unique(instance_mask).tolist())
    if not values <= allowed:
        raise ValueError(f"허용되지 않은 인스턴스 값입니다: {sorted(values - allowed)}")

    components: list[dict[str, object]] = []
    flags: list[str] = []
    for instance_id, part in (
        (HANDLE_INSTANCE_ID, "handle"),
        (BODY_INSTANCE_ID, "body"),
    ):
        binary = (instance_mask == instance_id).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = [
            int(stats[local_id, cv2.CC_STAT_AREA])
            for local_id in range(1, count)
            if int(stats[local_id, cv2.CC_STAT_AREA]) >= minimum_pixels
        ]
        pixel_count = int(np.count_nonzero(binary))
        if pixel_count:
            components.append(
                {
                    "instance_id": instance_id,
                    "part": part,
                    "label": "grasp_region",
                    "pixel_count": pixel_count,
                    "connected_component_count": len(areas),
                    "status": "final_confirmation_required",
                }
            )
        if part == "body" and pixel_count == 0:
            flags.append("몸통_누락")
        if part == "handle" and pixel_count == 0:
            # 손잡이가 카메라 반대편에 있으면 몸통만 보이는 것이 정상일 수 있다.
            flags.append("손잡이_비가시_또는_라벨없음")
        if len(areas) > 1:
            flags.append(f"{part}_다중_연결성분_확인")

    foreground = instance_mask > 0
    if np.any(foreground):
        if (
            np.any(foreground[0])
            or np.any(foreground[-1])
            or np.any(foreground[:, 0])
            or np.any(foreground[:, -1])
        ):
            flags.append("마스크_영상경계_접촉")
        if float(np.count_nonzero(foreground)) / foreground.size > 0.7:
            flags.append("마스크_면적_과다")
    return components, sorted(set(flags))


def render_review_overlay(image_bgr: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    """몸통은 청록색, 손잡이는 초록색으로 겹친 검수 이미지를 만든다."""

    if image_bgr.shape[:2] != instance_mask.shape:
        raise ValueError("원본 이미지와 인스턴스 마스크 크기가 다릅니다")
    result = image_bgr.copy()
    color = np.zeros_like(result)
    # OpenCV는 BGR 순서다. 기존 오버레이와 같은 색 의미를 유지한다.
    color[instance_mask == HANDLE_INSTANCE_ID] = (40, 200, 40)
    color[instance_mask == BODY_INSTANCE_ID] = (210, 190, 20)
    color[instance_mask == FUNCTIONAL_INSTANCE_ID] = (50, 50, 235)
    selected = instance_mask > 0
    # 불리언 인덱싱 결과가 픽셀 한 개뿐이면 일부 OpenCV 버전에서
    # addWeighted가 None을 반환한다. 전체 영상을 혼합한 뒤 필요한 픽셀만
    # 가져오면 작은 합성 테스트와 실제 고해상도 이미지 모두 안정적이다.
    blended = cv2.addWeighted(result, 0.45, color, 0.55, 0)
    result[selected] = blended[selected]
    return result


def _find_unique_raw_images(raw_root: Path) -> dict[str, Path]:
    """확장자를 포함한 파일명으로 원본을 찾고 중복 파일명은 거부한다."""

    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    grouped: dict[str, list[Path]] = {}
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            grouped.setdefault(path.name, []).append(path)
    duplicates = {name: paths for name, paths in grouped.items() if len(paths) > 1}
    if duplicates:
        names = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"원본 이미지 파일명이 중복됩니다: {names}")
    return {name: paths[0] for name, paths in grouped.items()}


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """JSONL을 UTF-8로 결정적으로 저장한다."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_image(path: Path, image: np.ndarray) -> None:
    """OpenCV 저장 실패를 조용히 넘기지 않고 대상 경로와 함께 보고한다."""

    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")


def prepare_review_workspace(
    review_config_path: Path,
    candidate_root: Path,
    raw_image_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """문제가 표시된 후보만 복사해 안전한 수동 수정 작업 공간을 만든다."""

    if output_root.exists():
        raise FileExistsError(f"기존 검수 작업 공간을 덮어쓰지 않습니다: {output_root}")

    reviews, config = load_review_notes(review_config_path)
    candidates = load_candidate_manifest(candidate_root / "candidate_manifest.jsonl")
    candidate_by_overlay = {
        Path(str(row["overlay_path"])).name: row for row in candidates
    }
    missing_candidates = sorted(set(reviews) - set(candidate_by_overlay))
    if missing_candidates:
        raise ValueError(
            "후보 manifest에서 찾지 못한 검수 파일: " + ", ".join(missing_candidates)
        )

    raw_by_name = _find_unique_raw_images(raw_image_root)
    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 검수 폴더를 먼저 확인하세요: {staging}")

    for directory in (
        "images",
        "candidate_instances",
        "corrected_instances",
        "corrected_semantic",
        "corrected_overlays",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, object]] = []
    for filename, note in reviews.items():
        candidate = candidate_by_overlay[filename]
        source_name = filename.split("__", maxsplit=1)[-1]
        raw_path = raw_by_name.get(source_name)
        if raw_path is None:
            raise FileNotFoundError(f"로컬 원본 이미지를 찾을 수 없습니다: {source_name}")

        candidate_instance_path = candidate_root / str(
            candidate["candidate_instance_path"]
        )
        candidate_mask = cv2.imread(str(candidate_instance_path), cv2.IMREAD_UNCHANGED)
        if candidate_mask is None:
            raise FileNotFoundError(f"후보 마스크를 읽을 수 없습니다: {candidate_instance_path}")
        standardized = standardize_mug_instances(
            candidate_mask, candidate.get("components", [])
        )

        target_image = staging / "images" / filename
        target_candidate = staging / "candidate_instances" / Path(filename).with_suffix(
            ".png"
        )
        target_corrected = staging / "corrected_instances" / target_candidate.name
        shutil.copy2(raw_path, target_image)
        _write_image(target_candidate, standardized)
        _write_image(target_corrected, standardized)

        review_rows.append(
            {
                "source_dataset": "custom",
                "source_id": candidate["source_id"],
                "object_id": candidate["object_id"],
                "split": candidate["split"],
                "image_path": target_image.relative_to(staging).as_posix(),
                "candidate_instance_path": target_candidate.relative_to(staging).as_posix(),
                "corrected_instance_path": target_corrected.relative_to(staging).as_posix(),
                "review_note": note,
                "review_status": "pending_correction",
                "functional_region_generated": False,
            }
        )

    _write_jsonl(staging / "review_manifest.jsonl", review_rows)
    issue_counts = Counter(str(row["review_note"]) for row in review_rows)
    summary: dict[str, object] = {
        "review_id": config.get("review_id"),
        "source_run": config.get("source_run"),
        "status": "human_correction_required",
        "items": len(review_rows),
        "by_object": dict(Counter(str(row["object_id"]) for row in review_rows)),
        "issue_counts": dict(sorted(issue_counts.items())),
        "fixed_instance_ids": {"1": "handle", "2": "body"},
        "final_semantic_label": "grasp_region",
        "functional_region_generated": False,
        "unlisted_status": config.get("unlisted_status"),
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "검수_수정_안내.txt").write_text(
        "손잡이는 초록색, 몸통은 청록색으로 표시됩니다.\n"
        "두 영역은 모두 최종 grasp_region이며 서로 다른 인스턴스로 유지합니다.\n"
        "원본 후보는 candidate_instances에 보존되고 수정은 corrected_instances에 저장됩니다.\n"
        "수정 화면 실행: python scripts/review_custom_masks.py\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    return summary


def prepare_candidate_review_workspace(
    candidate_root: Path,
    raw_image_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """반자동 후보 실행 전체를 수정 가능한 검수 작업 공간으로 만든다.

    2026-08-21 추가 촬영(mug_04·가위·드라이버) 검수용이다. 후보 실행의 모든
    이미지를 카테고리 정보와 함께 복사하고, 인스턴스 ID를 손잡이=1, 몸통=2,
    기능=3으로 표준화한다. 원본 후보는 절대 수정하지 않는다.
    """

    if output_root.exists():
        raise FileExistsError(f"기존 검수 작업 공간을 덮어쓰지 않습니다: {output_root}")
    candidates = load_candidate_manifest(candidate_root / "candidate_manifest.jsonl")
    if not candidates:
        raise ValueError("후보 manifest가 비어 있습니다")
    raw_by_name = _find_unique_raw_images(raw_image_root)

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 검수 폴더를 먼저 확인하세요: {staging}")
    for directory in (
        "images",
        "candidate_instances",
        "corrected_instances",
        "corrected_semantic",
        "corrected_overlays",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        filename = Path(str(candidate["overlay_path"])).name
        source_name = filename.split("__", maxsplit=1)[-1]
        raw_path = raw_by_name.get(source_name)
        if raw_path is None:
            raise FileNotFoundError(f"로컬 원본 이미지를 찾을 수 없습니다: {source_name}")
        candidate_instance_path = candidate_root / str(candidate["candidate_instance_path"])
        candidate_mask = cv2.imread(str(candidate_instance_path), cv2.IMREAD_UNCHANGED)
        if candidate_mask is None:
            raise FileNotFoundError(f"후보 마스크를 읽을 수 없습니다: {candidate_instance_path}")
        standardized = standardize_category_instances(
            candidate_mask, candidate.get("components", [])
        )

        mask_filename = Path(filename).with_suffix(".png")
        target_image = staging / "images" / filename
        target_candidate = staging / "candidate_instances" / mask_filename
        target_corrected = staging / "corrected_instances" / mask_filename
        shutil.copy2(raw_path, target_image)
        _write_image(target_candidate, standardized)
        _write_image(target_corrected, standardized)

        review_flags = [str(flag) for flag in candidate.get("review_flags", [])]
        review_note = (
            "신규 촬영 검수" if not review_flags
            else "신규 촬영 검수 / 자동 경고: " + ", ".join(review_flags)
        )
        rows.append(
            {
                "source_dataset": "custom",
                "source_id": candidate["source_id"],
                "object_id": candidate["object_id"],
                "category": candidate.get("category"),
                "split": candidate["split"],
                "image_path": target_image.relative_to(staging).as_posix(),
                "candidate_instance_path": target_candidate.relative_to(staging).as_posix(),
                "corrected_instance_path": target_corrected.relative_to(staging).as_posix(),
                "review_note": review_note,
                "review_status": "pending_correction",
                "handle_visibility": (
                    "visible"
                    if np.any(standardized == HANDLE_INSTANCE_ID)
                    else "not_visible"
                ),
                "functional_region_generated": bool(
                    np.any(standardized == FUNCTIONAL_INSTANCE_ID)
                ),
            }
        )

    _write_jsonl(staging / "review_manifest.jsonl", rows)
    summary: dict[str, object] = {
        "status": "human_correction_required",
        "items": len(rows),
        "by_object": dict(Counter(str(row["object_id"]) for row in rows)),
        "by_category": dict(Counter(str(row["category"]) for row in rows)),
        "by_split": dict(Counter(str(row["split"]) for row in rows)),
        "fixed_instance_ids": {"1": "handle", "2": "body", "3": "functional"},
        "source_candidate_root": candidate_root.as_posix(),
        "warning": "모든 후보는 사람 검수 전 초안이며 원본 후보 폴더는 변경하지 않습니다.",
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "검수_수정_안내.txt").write_text(
        "부위 색: 손잡이=초록(H), 몸통=청록(B, 머그만), 기능 부위=빨강(F, 가위 날·드라이버 축).\n"
        "머그: 손잡이/몸통만 사용하며 기능 부위를 만들지 않습니다.\n"
        "가위: 고리 두 개는 H, 날은 F로 칠합니다. 몸통(B)은 쓰지 않습니다.\n"
        "드라이버: 손잡이는 H, 축과 팁은 F로 칠합니다. 몸통(B)은 쓰지 않습니다.\n"
        "수정 화면 실행: python scripts/review_custom_masks.py --workspace <이 폴더>\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    return summary


def prepare_full_review_workspace(
    final_review_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """최종 확인 세트 전체를 수정 가능한 별도 작업 복사본으로 만든다.

    사용자가 필요한 사진만 고칠 수 있도록 140장 모두 현재 마스크로 초기화한다.
    입력 최종 세트와 이전 사람 수정본은 절대 덮어쓰지 않는다.
    """

    if output_root.exists():
        raise FileExistsError(f"기존 전체 수정 작업 공간을 덮어쓰지 않습니다: {output_root}")
    source_rows = load_candidate_manifest(
        final_review_root / "final_review_manifest.jsonl"
    )
    included_rows = [
        row
        for row in source_rows
        if row.get("instance_mask_path") and row.get("image_path")
    ]
    if not included_rows:
        raise ValueError("전체 수정 작업 공간으로 복사할 최종 마스크가 없습니다")

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"이전 임시 전체 수정 폴더를 먼저 확인하세요: {staging}")
    for directory in (
        "images",
        "candidate_instances",
        "corrected_instances",
        "corrected_semantic",
        "corrected_overlays",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in included_rows:
        source_image = final_review_root / str(source["image_path"])
        source_mask = final_review_root / str(source["instance_mask_path"])
        if not source_image.is_file():
            raise FileNotFoundError(f"최종 원본 이미지를 찾을 수 없습니다: {source_image}")
        mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"최종 인스턴스 마스크를 읽을 수 없습니다: {source_mask}")
        values = set(np.unique(mask).tolist())
        if not values <= {0, HANDLE_INSTANCE_ID, BODY_INSTANCE_ID}:
            raise ValueError(
                f"전체 수정 대상에 허용되지 않은 마스크 값이 있습니다: {source['source_id']}"
            )

        filename = Path(str(source["image_path"])).name
        mask_filename = Path(filename).with_suffix(".png")
        target_image = staging / "images" / filename
        target_candidate = staging / "candidate_instances" / mask_filename
        target_corrected = staging / "corrected_instances" / mask_filename
        shutil.copy2(source_image, target_image)
        _write_image(target_candidate, mask)
        _write_image(target_corrected, mask)
        validation_flags = [str(flag) for flag in source.get("validation_flags", [])]
        review_note = (
            "전체 최종 검수"
            if not validation_flags
            else "전체 최종 검수 / 자동 경고: " + ", ".join(validation_flags)
        )
        rows.append(
            {
                "source_dataset": "custom",
                "source_id": source["source_id"],
                "object_id": source["object_id"],
                "split": source["split"],
                "image_path": target_image.relative_to(staging).as_posix(),
                "candidate_instance_path": target_candidate.relative_to(staging).as_posix(),
                "corrected_instance_path": target_corrected.relative_to(staging).as_posix(),
                "review_note": review_note,
                "review_status": "ready_for_optional_edit",
                "handle_visibility": (
                    "visible"
                    if np.any(mask == HANDLE_INSTANCE_ID)
                    else "not_visible"
                ),
                "source_label_status": source.get("annotation_status"),
                "source_label_kind": source.get("label_source"),
                "functional_region_generated": False,
            }
        )

    _write_jsonl(staging / "review_manifest.jsonl", rows)
    summary: dict[str, object] = {
        "status": "ready_for_optional_edit",
        "items": len(rows),
        "by_object": dict(Counter(str(row["object_id"]) for row in rows)),
        "by_split": dict(Counter(str(row["split"]) for row in rows)),
        "fixed_instance_ids": {"1": "handle", "2": "body"},
        "final_semantic_label": "grasp_region",
        "functional_region_generated": False,
        "source_final_review_root": final_review_root.as_posix(),
        "warning": "필요한 사진만 수정하며 원본 최종 확인 세트는 변경하지 않습니다.",
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "전체_수정_안내.txt").write_text(
        "전체 140장의 현재 마스크를 수정 가능한 복사본으로 준비했습니다.\n"
        "N/P로 이동하고 이상한 사진만 수정한 뒤 A로 저장하세요.\n"
        "정상 사진은 N으로 넘겨도 기존 마스크가 그대로 보존됩니다.\n"
        "손잡이가 실제로 안 보이면 손잡이 픽셀을 지운 후 V를 켜고 A로 저장하세요.\n"
        "모든 작업을 끝낸 뒤 Q로 종료하고 Codex에 전체 수정 완료라고 알려주세요.\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    return summary
