"""자체 촬영 이미지의 반자동 라벨 후보를 안전하게 조립하는 도우미."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


GRASP_SEMANTIC_VALUE = 1
FUNCTIONAL_SEMANTIC_VALUE = 2
IGNORE_VALUE = 255


@dataclass(frozen=True)
class MaskCandidate:
    """Grounding DINO의 상자와 SAM2 마스크가 만든 한 개의 검수 후보."""

    part: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray


def select_even_samples(
    records: Iterable[dict[str, object]],
    sample_per_object: int | None,
) -> list[dict[str, object]]:
    """각 실제 물체에서 시간순으로 고르게 샘플을 선택한다.

    프레임 단위 무작위 선택을 피하고, 촬영 구간의 앞·중간·뒤가 포함되도록 한다.
    ``sample_per_object``가 None이면 모든 라벨링 대상 이미지를 반환한다.
    """

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if bool(record.get("is_negative", False)):
            continue
        if record.get("conversion_status") != "pending_annotation":
            continue
        grouped[str(record["object_id"])].append(record)

    selected: list[dict[str, object]] = []
    for object_id in sorted(grouped):
        rows = sorted(grouped[object_id], key=lambda row: str(row["source_id"]))
        if sample_per_object is None or sample_per_object >= len(rows):
            selected.extend(rows)
            continue
        if sample_per_object < 1:
            raise ValueError("sample_per_object는 1 이상이어야 합니다")

        # linspace를 이용해 첫·마지막 프레임을 포함하면서 중복 없는 위치를 고른다.
        indices = np.linspace(0, len(rows) - 1, sample_per_object, dtype=int)
        selected.extend(rows[int(index)] for index in indices)
    return selected


def combine_grasp_candidates(
    candidates: Sequence[MaskCandidate],
    image_shape: tuple[int, int],
    *,
    minimum_pixels: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    """손잡이와 몸통 후보를 별도 인스턴스로 유지하며 마스크로 합친다.

    손잡이 후보를 먼저 배치하고 몸통에서 겹친 픽셀을 제거한다. 따라서 물리적으로
    맞닿아 있어도 instance mask에서는 서로 다른 ID를 유지한다. 이 함수는 머그에
    대해 ``functional_region``을 만들지 않는다.
    """

    height, width = image_shape
    semantic = np.zeros((height, width), dtype=np.uint8)
    instance = np.zeros((height, width), dtype=np.uint16)
    components: list[dict[str, object]] = []
    flags: list[str] = []

    parts = {candidate.part for candidate in candidates}
    if "handle" not in parts:
        flags.append("손잡이_후보_누락")
    if "body" not in parts:
        flags.append("몸통_후보_누락")

    # 같은 부위 후보는 신뢰도가 높은 것부터 처리하며 손잡이를 몸통보다 우선한다.
    ordered = sorted(
        candidates,
        key=lambda candidate: (0 if candidate.part == "handle" else 1, -candidate.score),
    )
    occupied = np.zeros((height, width), dtype=bool)
    next_instance_id = 1
    for candidate in ordered:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"후보 마스크 크기가 이미지와 다릅니다: {mask.shape} != {(height, width)}"
            )
        separated = mask & ~occupied
        pixel_count = int(separated.sum())
        if pixel_count < minimum_pixels:
            flags.append(f"너무_작은_{candidate.part}_후보")
            continue

        semantic[separated] = GRASP_SEMANTIC_VALUE
        instance[separated] = next_instance_id
        occupied |= separated
        components.append(
            {
                "instance_id": next_instance_id,
                "part": candidate.part,
                "label": "grasp_region",
                "score": round(float(candidate.score), 6),
                "box_xyxy": [round(float(value), 2) for value in candidate.box_xyxy],
                "pixel_count": pixel_count,
                "status": "human_review_required",
            }
        )
        next_instance_id += 1

    if not components:
        flags.append("유효한_마스크_없음")
    if FUNCTIONAL_SEMANTIC_VALUE in semantic:
        raise AssertionError("머그 반자동 후보에 functional_region이 생성되었습니다")
    return semantic, instance, components, sorted(set(flags))


def _largest_components(mask: np.ndarray, minimum_pixels: int) -> list[np.ndarray]:
    """작은 경계 잡음을 버리고 남은 연결 성분을 큰 순서로 반환한다."""

    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= minimum_pixels:
            components.append((area, labels == label_id))
    return [component for _, component in sorted(components, reverse=True, key=lambda row: row[0])]


def split_whole_mug_mask(
    whole_mask: np.ndarray,
    *,
    minimum_handle_pixels: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """컵 전체 마스크의 중심 몸통과 옆으로 돌출된 손잡이를 기하적으로 분리한다.

    위에서 본 둥근 컵은 최대 내접점 기준 타원을 사용하고, 옆에서 본 컵은 행별
    중심 구간의 좌우 경계를 중앙값으로 평활화한다. 결과는 어디까지나 사람 검수용
    후보이며, 실패하거나 비율이 비정상적이면 검수 표시를 남긴다.
    """

    import cv2

    whole = np.asarray(whole_mask, dtype=bool)
    if whole.ndim != 2 or not whole.any():
        raise ValueError("컵 전체 마스크가 비어 있거나 2차원이 아닙니다")
    ys, xs = np.where(whole)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_height = y_max - y_min + 1
    distance = cv2.distanceTransform(whole.astype(np.uint8), cv2.DIST_L2, 5)
    center_y, center_x = np.unravel_index(int(distance.argmax()), distance.shape)
    radius_ratio = float(distance.max()) / float(bbox_height)
    flags: list[str] = []

    if radius_ratio >= 0.42:
        method = "top_view_ellipse"

        def distance_until_background(dx: int, dy: int) -> int:
            step = 0
            while True:
                x = center_x + dx * step
                y = center_y + dy * step
                if not (0 <= x < whole.shape[1] and 0 <= y < whole.shape[0]):
                    return max(1, step - 1)
                if not whole[y, x]:
                    return max(1, step - 1)
                step += 1

        # 손잡이가 한쪽 반지름을 늘려도 반대쪽과의 최솟값을 써서 몸통 크기를 지킨다.
        radius_x = min(
            distance_until_background(-1, 0), distance_until_background(1, 0)
        )
        radius_y = min(
            distance_until_background(0, -1), distance_until_background(0, 1)
        )
        yy, xx = np.ogrid[: whole.shape[0], : whole.shape[1]]
        ellipse = (
            ((xx - center_x) / max(1.0, radius_x * 1.05)) ** 2
            + ((yy - center_y) / max(1.0, radius_y * 1.05)) ** 2
            <= 1.0
        )
        body_seed = whole & ellipse
    else:
        method = "side_view_row_envelope"
        left_bounds = np.full(whole.shape[0], np.nan, dtype=np.float64)
        right_bounds = np.full(whole.shape[0], np.nan, dtype=np.float64)
        for y in range(y_min, y_max + 1):
            row_x = np.flatnonzero(whole[y])
            if row_x.size == 0:
                continue
            gaps = np.flatnonzero(np.diff(row_x) > 1)
            starts = np.r_[0, gaps + 1]
            ends = np.r_[gaps, row_x.size - 1]
            runs = [(int(row_x[start]), int(row_x[end])) for start, end in zip(starts, ends)]
            containing = [run for run in runs if run[0] <= center_x <= run[1]]
            chosen = containing[0] if containing else max(runs, key=lambda run: run[1] - run[0])
            left_bounds[y], right_bounds[y] = chosen

        window = max(5, int(round(bbox_height * 0.11)))
        if window % 2 == 0:
            window += 1
        half = window // 2
        smooth_left = left_bounds.copy()
        smooth_right = right_bounds.copy()
        for y in range(y_min, y_max + 1):
            low, high = max(y_min, y - half), min(y_max + 1, y + half + 1)
            valid_left = left_bounds[low:high]
            valid_right = right_bounds[low:high]
            if np.isfinite(valid_left).any():
                smooth_left[y] = np.nanmedian(valid_left)
                smooth_right[y] = np.nanmedian(valid_right)
        body_seed = np.zeros_like(whole)
        for y in range(y_min, y_max + 1):
            if not np.isfinite(smooth_left[y]):
                continue
            left = max(0, int(round(smooth_left[y])) - 2)
            right = min(whole.shape[1], int(round(smooth_right[y])) + 3)
            body_seed[y, left:right] = whole[y, left:right]

    residual = whole & ~body_seed
    residual_components = _largest_components(residual, minimum_handle_pixels)
    if residual_components:
        # 중심에서 수평으로 가장 멀고 충분히 큰 성분을 손잡이로 선택한다.
        handle = max(
            residual_components,
            key=lambda component: (
                abs(float(np.where(component)[1].mean()) - center_x),
                int(component.sum()),
            ),
        )
    else:
        handle = np.zeros_like(whole)
        flags.append("형태분석_손잡이_누락")
    body = whole & ~handle

    whole_pixels = int(whole.sum())
    handle_ratio = float(handle.sum()) / whole_pixels
    body_ratio = float(body.sum()) / whole_pixels
    if handle.any() and handle_ratio < 0.01:
        flags.append("형태분석_손잡이_면적_과소")
    if handle_ratio > 0.45:
        flags.append("형태분석_손잡이_면적_과다")
    if body_ratio < 0.50:
        flags.append("형태분석_몸통_면적_과소")
    diagnostics = {
        "method": method,
        "center_xy": [int(center_x), int(center_y)],
        "radius_ratio": round(radius_ratio, 6),
        "whole_pixels": whole_pixels,
        "body_pixels": int(body.sum()),
        "handle_pixels": int(handle.sum()),
        "handle_ratio": round(handle_ratio, 6),
    }
    return body, handle, sorted(set(flags)), diagnostics


def split_anchor_residual(
    whole_mask: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    max_anchor_instances: int,
    minimum_pixels: int = 64,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, object]]:
    """전체 마스크에서 앵커 부위(손잡이)를 빼 잔여 부위를 만든다.

    2026-08-21 승인된 가위·드라이버 부위 정책용 분리기다.

    - 앵커(손잡이) 마스크는 전체 마스크와 교집합으로 제한한 뒤 연결 성분으로
      나눠 최대 ``max_anchor_instances``개 인스턴스로 유지한다(가위 고리 2개).
    - 잔여(전체 - 앵커)는 기능 부위(가위 날, 드라이버 축) 후보가 된다.
    - 어느 쪽이 비면 검수 표시를 남긴다. 결과는 사람 검수 전 후보일 뿐이다.
    """

    whole = np.asarray(whole_mask, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool) & whole
    if whole.ndim != 2:
        raise ValueError("전체 마스크는 2차원이어야 합니다")
    if max_anchor_instances < 1:
        raise ValueError(f"max_anchor_instances는 1 이상이어야 합니다: {max_anchor_instances}")

    flags: list[str] = []
    anchor_components = _largest_components(anchor, minimum_pixels)[:max_anchor_instances]
    if not anchor_components:
        flags.append("앵커_부위_후보_누락")
    elif len(anchor_components) < max_anchor_instances:
        flags.append("앵커_인스턴스_부족")

    anchor_union = np.zeros_like(whole)
    for component in anchor_components:
        anchor_union |= component
    residual_raw = whole & ~anchor_union
    residual_components = _largest_components(residual_raw, minimum_pixels)
    residual = np.zeros_like(whole)
    for component in residual_components:
        residual |= component
    if not residual.any():
        flags.append("잔여_부위_후보_누락")

    whole_pixels = int(whole.sum())
    diagnostics = {
        "whole_pixels": whole_pixels,
        "anchor_pixels": int(anchor_union.sum()),
        "residual_pixels": int(residual.sum()),
        "anchor_instances": len(anchor_components),
        "anchor_ratio": round(float(anchor_union.sum()) / whole_pixels, 6) if whole_pixels else None,
    }
    return anchor_components, residual, sorted(set(flags)), diagnostics


def _principal_axis_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """마스크 픽셀의 주축 좌표 t(길이 방향)와 s(수직 방향)를 반환한다."""

    ys, xs = np.where(mask)
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = points.mean(axis=0)
    centered = points - center
    # SVD의 첫 특이벡터가 길이 방향 주축이다.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis, normal = vt[0], vt[1]
    return centered @ axis, centered @ normal


def split_scissors_by_rings(
    whole_mask: np.ndarray,
    *,
    minimum_pixels: int = 64,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, object]]:
    """가위 전체 마스크를 고리(손잡이)와 날(기능 부위)로 기하학적으로 나눈다.

    손잡이 고리는 마스크 내부의 구멍(배경으로 닫힌 영역)으로 찾는다. 구멍들이
    있는 주축 구간까지를 손잡이로, 나머지를 날로 본다. DINO 부위 프롬프트가
    가위 전체를 반환하는 문제(2026-08-21 확인) 때문에 도입한 분리기이며,
    결과는 사람 검수용 후보일 뿐이다.
    """

    import cv2

    whole = np.asarray(whole_mask, dtype=bool)
    if whole.ndim != 2 or not whole.any():
        raise ValueError("가위 전체 마스크가 비어 있거나 2차원이 아닙니다")
    flags: list[str] = []

    # 테두리에서 배경을 채워 남는 미충전 배경 = 마스크 내부 구멍(고리 안쪽)
    background = (~whole).astype(np.uint8)
    flood = background.copy()
    flood_mask = np.zeros((whole.shape[0] + 2, whole.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    holes = flood.astype(bool)
    hole_components = _largest_components(holes, max(16, minimum_pixels // 4))[:2]

    if not hole_components:
        flags.append("가위_고리_구멍_미검출")
        return [], whole.copy(), sorted(set(flags)), {
            "whole_pixels": int(whole.sum()),
            "hole_count": 0,
        }
    if len(hole_components) < 2:
        flags.append("가위_고리_구멍_하나만_검출")

    # 주축은 한 번만 계산해 마스크와 구멍이 같은 좌표계를 쓰게 한다.
    ys, xs = np.where(whole)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axis = vt[0]
    t_all = (points - center) @ axis
    t_image = np.zeros(whole.shape, dtype=np.float64)
    t_image[ys, xs] = t_all

    hole_ts: list[float] = []
    for hole in hole_components:
        hys, hxs = np.where(hole)
        hole_points = np.stack([hxs, hys], axis=1).astype(np.float64)
        hole_ts.extend(((hole_points - center) @ axis).tolist())
    hole_ts_array = np.asarray(hole_ts)

    # 구멍이 주축의 음수 쪽에 오도록 방향을 정한다.
    if float(np.median(hole_ts_array)) > 0:
        t_all = -t_all
        t_image = -t_image
        hole_ts_array = -hole_ts_array
    span = float(t_all.max() - t_all.min())
    cut = float(hole_ts_array.max()) + span * 0.04
    handle_zone = whole & (t_image <= cut)
    blade_zone = whole & (t_image > cut)

    ring_components = _largest_components(handle_zone, minimum_pixels)[:2]
    blade_components = _largest_components(blade_zone, minimum_pixels)
    blade = np.zeros_like(whole)
    for component in blade_components:
        blade |= component
    if not ring_components:
        flags.append("가위_손잡이_구간_비어있음")
    if not blade.any():
        flags.append("가위_날_구간_비어있음")

    whole_pixels = int(whole.sum())
    diagnostics = {
        "whole_pixels": whole_pixels,
        "hole_count": len(hole_components),
        "handle_pixels": int(handle_zone.sum()),
        "blade_pixels": int(blade.sum()),
        "handle_ratio": round(float(handle_zone.sum()) / whole_pixels, 6),
        "cut_position": round(cut, 2),
    }
    return ring_components, blade, sorted(set(flags)), diagnostics


def split_screwdriver_by_width(
    whole_mask: np.ndarray,
    *,
    minimum_pixels: int = 64,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, object]]:
    """드라이버 전체 마스크를 폭 변화로 손잡이와 축(기능 부위)으로 나눈다.

    주축을 따라 구간별 수직 폭을 재고, 두꺼운 쪽(손잡이)의 중앙값 폭 대비
    절반 아래로 떨어지는 지점을 경계로 삼는다. 결과는 사람 검수용 후보다.
    """

    whole = np.asarray(whole_mask, dtype=bool)
    if whole.ndim != 2 or not whole.any():
        raise ValueError("드라이버 전체 마스크가 비어 있거나 2차원이 아닙니다")
    flags: list[str] = []

    t_all, s_all = _principal_axis_coordinates(whole)
    bin_count = 48
    edges = np.linspace(float(t_all.min()), float(t_all.max()), bin_count + 1)
    widths = np.zeros(bin_count, dtype=np.float64)
    for index in range(bin_count):
        inside = (t_all >= edges[index]) & (t_all <= edges[index + 1])
        if inside.any():
            widths[index] = float(s_all[inside].max() - s_all[inside].min())

    quarter = max(4, bin_count // 4)
    # 손잡이(두꺼운 끝)가 낮은 t 쪽에 오도록 방향을 정한다.
    if float(np.median(widths[-quarter:])) > float(np.median(widths[:quarter])):
        t_all = -t_all
        widths = widths[::-1]
        edges = -edges[::-1]
    handle_width = float(np.median(widths[:quarter][widths[:quarter] > 0]))

    cut_index = None
    for index in range(quarter, bin_count - 1):
        if widths[index] > 0 and widths[index] < handle_width * 0.5:
            if index + 1 < bin_count and 0 < widths[index + 1] < handle_width * 0.5:
                cut_index = index
                break
    if cut_index is None:
        flags.append("드라이버_폭_경계_미검출")
        return [], whole.copy(), sorted(set(flags)), {
            "whole_pixels": int(whole.sum()),
            "handle_width": round(handle_width, 2),
        }

    cut = float(edges[cut_index])
    ys, xs = np.where(whole)
    t_image = np.zeros(whole.shape, dtype=np.float64)
    t_image[ys, xs] = t_all
    handle_zone = whole & (t_image <= cut)
    shaft_zone = whole & (t_image > cut)

    handle_components = _largest_components(handle_zone, minimum_pixels)[:1]
    shaft_components = _largest_components(shaft_zone, minimum_pixels)
    shaft = np.zeros_like(whole)
    for component in shaft_components:
        shaft |= component
    if not handle_components:
        flags.append("드라이버_손잡이_구간_비어있음")
    if not shaft.any():
        flags.append("드라이버_축_구간_비어있음")

    whole_pixels = int(whole.sum())
    diagnostics = {
        "whole_pixels": whole_pixels,
        "handle_pixels": int(handle_zone.sum()),
        "shaft_pixels": int(shaft.sum()),
        "handle_width": round(handle_width, 2),
        "handle_ratio": round(float(handle_zone.sum()) / whole_pixels, 6),
    }
    return handle_components, shaft, sorted(set(flags)), diagnostics


def combine_category_candidates(
    anchor_components: Sequence[np.ndarray],
    residual_mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    anchor_part: str,
    residual_part: str,
    residual_label: str,
    score: float,
    box_xyxy: tuple[float, float, float, float],
    minimum_pixels: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    """앵커·잔여 부위를 semantic/instance 마스크와 검수 컴포넌트로 조립한다.

    앵커 부위는 항상 ``grasp_region``(저장값 1)이고, 잔여 부위는
    ``residual_label``에 따라 ``grasp_region`` 또는 ``functional_region``
    (저장값 2)이 된다. 모든 부위는 별도 인스턴스 ID를 유지한다.
    """

    if residual_label not in ("grasp_region", "functional_region"):
        raise ValueError(f"허용되지 않는 잔여 라벨입니다: {residual_label}")
    height, width = image_shape
    semantic = np.zeros((height, width), dtype=np.uint8)
    instance = np.zeros((height, width), dtype=np.uint16)
    components: list[dict[str, object]] = []
    flags: list[str] = []
    occupied = np.zeros((height, width), dtype=bool)
    next_instance_id = 1

    def add_component(mask: np.ndarray, part: str, label: str) -> None:
        nonlocal next_instance_id
        separated = np.asarray(mask, dtype=bool) & ~occupied
        pixel_count = int(separated.sum())
        if pixel_count < minimum_pixels:
            flags.append(f"너무_작은_{part}_후보")
            return
        value = GRASP_SEMANTIC_VALUE if label == "grasp_region" else FUNCTIONAL_SEMANTIC_VALUE
        semantic[separated] = value
        instance[separated] = next_instance_id
        occupied[separated] = True
        components.append(
            {
                "instance_id": next_instance_id,
                "part": part,
                "label": label,
                "score": round(float(score), 6),
                "box_xyxy": [round(float(v), 2) for v in box_xyxy],
                "pixel_count": pixel_count,
                "status": "human_review_required",
            }
        )
        next_instance_id += 1

    for component in anchor_components:
        add_component(component, anchor_part, "grasp_region")
    add_component(residual_mask, residual_part, residual_label)

    if not components:
        flags.append("유효한_마스크_없음")
    if not any(row["label"] == "grasp_region" for row in components):
        flags.append("파지_부위_없음")
    return semantic, instance, components, sorted(set(flags))


def resolve_repository_path(repository_root: Path, value: object) -> Path:
    """manifest의 상대경로를 저장소 루트 기준 실제 경로로 바꾼다."""

    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path
