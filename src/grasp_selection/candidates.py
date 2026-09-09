"""분할 결과에서 파지 후보를 추출한다.

마스크 centroid는 경계 밖이나 가장자리에 찍힐 수 있으므로 사용하지 않고,
distance transform의 최댓값 지점을 안전한 내부 파지점으로 쓴다. 그 최댓값
(내접원 반지름)의 2배를 후보 폭 추정치로 사용한다 — 손가락이 실제로 닿는
국소 폭에 해당하는 보수적인 추정이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MODEL_CLASS_NAMES = {0: "handle_grasp_region", 1: "body_grasp_region", 2: "functional_region"}


@dataclass
class GraspCandidate:
    """파지 후보 하나 (기능 영역은 후보가 아니라 회피 대상으로 별도 보관)."""

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray                     # bool (H, W)
    grasp_point: tuple[int, int]         # (x, y) 안전 내부점
    width_px: float                      # 내접원 지름 기반 폭 추정
    area_px: int
    touches_boundary: bool
    scores: dict[str, float] = field(default_factory=dict)  # 채점 단계에서 기록


def _instance_masks(result, frame_shape) -> list[tuple[int, float, np.ndarray]]:
    """Ultralytics 결과에서 (클래스, 신뢰도, bool 마스크) 목록을 만든다."""

    output = []
    if result.masks is None or result.boxes is None:
        return output
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    for mask, class_id, confidence in zip(result.masks.data.cpu().numpy(), classes, confidences):
        if mask.shape != frame_shape[:2]:
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        output.append((int(class_id), float(confidence), mask > 0.5))
    return output


def _array_masks(masks, class_ids, confidences, frame_shape) -> list[tuple[int, float, np.ndarray]]:
    """마스크 배열에서 (클래스, 신뢰도, bool 마스크) 목록을 만든다.

    RF-DETR(supervision Detections)처럼 Ultralytics 형식이 아닌 결과를 받기 위한
    경로다. masks는 (N, H, W), class_ids/confidences는 길이 N이다.
    """

    output = []
    if masks is None or len(masks) == 0:
        return output
    height, width = frame_shape[:2]
    for mask, class_id, confidence in zip(
        np.asarray(masks), np.asarray(class_ids, dtype=int), np.asarray(confidences, dtype=float)
    ):
        mask = np.asarray(mask)
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            )
        output.append((int(class_id), float(confidence), mask > 0.5))
    return output


def extract_candidates(
    result,
    frame_shape,
    *,
    min_confidence: float = 0.35,
    min_area_px: int = 400,
    boundary_margin_px: int = 4,
) -> tuple[list[GraspCandidate], np.ndarray]:
    """Ultralytics 분할 결과에서 파지 후보 목록과 기능 영역 마스크를 만든다."""

    return _build_candidates(
        _instance_masks(result, frame_shape),
        frame_shape,
        min_confidence=min_confidence,
        min_area_px=min_area_px,
        boundary_margin_px=boundary_margin_px,
    )


def extract_candidates_from_arrays(
    masks,
    class_ids,
    confidences,
    frame_shape,
    *,
    min_confidence: float = 0.35,
    min_area_px: int = 400,
    boundary_margin_px: int = 4,
) -> tuple[list[GraspCandidate], np.ndarray]:
    """마스크 배열에서 후보를 만든다(RF-DETR 등 비-Ultralytics 경로).

    클래스 번호 규약은 Ultralytics 경로와 같다: 0 handle, 1 body, 2 functional.
    호출 측에서 모델의 class_id 기준(1부터 시작하는 경우 등)을 먼저 맞춰야 한다.
    """

    return _build_candidates(
        _array_masks(masks, class_ids, confidences, frame_shape),
        frame_shape,
        min_confidence=min_confidence,
        min_area_px=min_area_px,
        boundary_margin_px=boundary_margin_px,
    )


def _build_candidates(
    instances: list[tuple[int, float, np.ndarray]],
    frame_shape,
    *,
    min_confidence: float,
    min_area_px: int,
    boundary_margin_px: int,
) -> tuple[list[GraspCandidate], np.ndarray]:
    """(클래스, 신뢰도, 마스크) 목록에서 후보와 기능 영역 마스크를 만든다.

    handle/body 인스턴스만 후보가 되고, functional 인스턴스는 회피 계산용
    통합 마스크로 합친다. 같은 클래스의 분리 인스턴스는 각각 후보로 유지한다.
    """

    height, width = frame_shape[:2]
    candidates: list[GraspCandidate] = []
    functional_mask = np.zeros((height, width), dtype=bool)

    for class_id, confidence, mask in instances:
        if class_id == 2:
            # 기능 영역은 신뢰도 조건 없이 보수적으로 회피 대상에 포함한다.
            functional_mask |= mask
            continue
        if confidence < min_confidence:
            continue
        area = int(mask.sum())
        if area < min_area_px:
            continue
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        _, max_value, _, max_location = cv2.minMaxLoc(distance)
        if max_value <= 0:
            continue
        edge = np.zeros_like(mask)
        margin = boundary_margin_px
        edge[:margin, :] = True
        edge[-margin:, :] = True
        edge[:, :margin] = True
        edge[:, -margin:] = True
        candidates.append(
            GraspCandidate(
                class_id=class_id,
                class_name=MODEL_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                confidence=confidence,
                mask=mask,
                grasp_point=(int(max_location[0]), int(max_location[1])),
                width_px=float(2.0 * max_value),
                area_px=area,
                touches_boundary=bool((mask & edge).any()),
            )
        )
    # 같은 클래스에서 서로 크게 겹치는 인스턴스(중복 검출·부분 검출)는
    # 신뢰도가 높은 쪽 하나로 병합해 점수 파편화를 막는다. 손잡이/몸통처럼
    # 클래스가 다른 후보는 절대 병합하지 않는다.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    kept: list[GraspCandidate] = []
    for candidate in candidates:
        duplicated = False
        for existing in kept:
            if existing.class_id != candidate.class_id:
                continue
            overlap = int((existing.mask & candidate.mask).sum())
            if overlap / max(candidate.area_px, 1) > 0.6:
                duplicated = True
                break
        if not duplicated:
            kept.append(candidate)
    return kept, functional_mask
