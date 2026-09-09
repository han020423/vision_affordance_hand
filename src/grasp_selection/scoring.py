"""파지 후보 채점과 자세 결정.

점수 = w_C·신뢰도 + w_D·거리 + w_A·접근방향 정렬 + w_F·폭 적합
     + w_M·기능영역 안전거리 + w_H·인간 파지 사전확률

자세(PRECISION/WRAP/POWER)는 폭 임계값이 아니라 물체 구조로 결정한다
(_decide_pose 참고; 근거는 VISOR 868건 기하 분석, docs/grasp_pose_geometry_analysis.md).

모든 항은 0~1로 정규화한다. 가중치·임계값은 configs/grasp_selection.yaml,
인간 사전확률은 configs/human_grasp_prior.yaml에서 읽는다. 신뢰도가 낮거나
1·2위 점수 차가 작으면 자동 파지 대신 ALIGN을 반환한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .candidates import GraspCandidate


@dataclass
class HandState:
    """손(또는 모의 손)의 화면 좌표 상태."""

    position: tuple[float, float]                 # (x, y) px
    direction: tuple[float, float] | None = None  # 접근 방향 단위벡터 (없으면 방향 항 중립)
    mm_per_px: float | None = None                # 손 폭 실측 기반 스케일 (없으면 설정값 사용)


@dataclass
class Decision:
    """후보 선택 결과."""

    state: str                                  # "GRASP" | "ALIGN" | "NO_TARGET"
    candidate: GraspCandidate | None = None
    pose: str | None = None                     # "PRECISION" | "WRAP" | "POWER"
    reason: str = ""
    ranked: list[GraspCandidate] = field(default_factory=list)


def load_selection_config(config_path: Path, prior_path: Path) -> dict:
    """선택 설정과 인간 사전확률을 읽어 하나의 딕셔너리로 합친다."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prior = yaml.safe_load(prior_path.read_text(encoding="utf-8"))
    config["human_prior"] = dict(prior["class_prior"])
    weight_sum = sum(config["weights"].values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"점수 가중치의 합이 1이 아닙니다: {weight_sum}")
    return config


def _width_fitness(candidate: GraspCandidate, pose_cfg: dict) -> tuple[float, bool]:
    """폭 적합도(0~1)와 최대 개구 초과 여부를 계산한다.

    화면 픽셀 폭은 카메라 거리로 인해 실제 폭과 비례하지 않으므로,
    mm 캘리브레이션(mm_per_px) 전에는 폭 적합도를 중립(0.5)으로 두고
    최대 개구 거부도 하지 않는다. 자세 결정은 폭이 아니라 _decide_pose의
    구조 규칙을 쓴다(2026-08-24 VISOR 분석: 폭은 파지 유형을 가르지 못함).
    """

    mm_per_px = pose_cfg.get("mm_per_px")
    if mm_per_px:
        width = candidate.width_px * float(mm_per_px)
        grasp_max = float(pose_cfg["max_grasp_width_mm"])
        calibrated = True
    else:
        width = candidate.width_px
        grasp_max = float(pose_cfg["max_grasp_width_px"])
        calibrated = False

    too_wide = calibrated and width > grasp_max

    if not calibrated:
        return 0.5, too_wide
    # 자세별 이상 폭에서 멀수록 감점하는 삼각형 곡선 (실측 기반일 때만).
    ideal_ratio = dict(pose_cfg.get("ideal_width_ratio", {}))
    ratio = float(ideal_ratio.get(candidate.class_name, 0.4))
    ideal = ratio * grasp_max
    fitness = float(np.clip(1.0 - abs(width - ideal) / grasp_max, 0.0, 1.0)) if grasp_max > 0 else 0.0
    if width < 6:
        fitness *= 0.5
    return fitness, too_wide


def _rect_sides(mask: np.ndarray) -> tuple[float, float]:
    """마스크 최소면적사각형의 (짧은 변, 긴 변)."""

    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0, 0.0
    points = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    return float(min(w, h)), float(max(w, h))


def _decide_pose(
    candidate: GraspCandidate,
    body_masks: list[np.ndarray],
    functional_mask: np.ndarray | None,
    pose_cfg: dict,
) -> str:
    """파지 자세 결정 (VISOR 868건 기하 분석 근거; docs/grasp_pose_geometry_analysis.md).

    사람 파지 측정에서 "잡은 부위의 폭"은 유형을 가르지 못했고(손바닥 폭의
    15~19%로 거의 일정, 균형정확도 0.50), 물체의 구조가 갈랐다. 규칙:

    1) 몸통(body) 후보 → POWER.
    2) 손잡이 후보가 몸통 영역과 인접 → 용기의 부속 손잡이 → WRAP
       (VISOR 부속성 특징 경계 0.27, 균형정확도 0.805).
    3) 그 외에는 물체 문맥(손잡이 ∪ 인접 기능영역)의 종횡비로 도구성 판단:
       길쭉하면(≥ aspect_tool_min; VISOR 경계 2.12, 균형정확도 0.936)
       도구 자루/펜 → PRECISION, 둥글면 → WRAP.

    모든 특징이 비율·인접성이라 카메라 거리와 mm 보정에 무관하다.
    """

    if candidate.class_id == 1:
        return "POWER"

    # 인접성 판정 반경은 후보 크기에 비례시켜 축척 불변으로 만든다.
    radius = max(5, int(round(0.15 * math.sqrt(max(candidate.area_px, 1)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(candidate.mask.astype(np.uint8), kernel).astype(bool)

    adjacency_min = int(pose_cfg.get("body_adjacency_min_px", 20))
    for body_mask in body_masks:
        if body_mask is candidate.mask:
            continue
        if int((dilated & body_mask).sum()) >= adjacency_min:
            return "WRAP"

    context = candidate.mask.copy()
    if functional_mask is not None and functional_mask.any():
        # 인접한 기능 영역은 연결 성분 전체를 문맥에 포함한다 (가위 날, 드라이버 축 등).
        _, labels = cv2.connectedComponents(functional_mask.astype(np.uint8))
        for label in set(np.unique(labels[dilated])) - {0}:
            context |= labels == label
    minor, major = _rect_sides(context)
    aspect = (major / minor) if minor > 0 else 1.0
    return "PRECISION" if aspect >= float(pose_cfg.get("aspect_tool_min", 2.1)) else "WRAP"


def _parts_adjacent(a: GraspCandidate, b: GraspCandidate, pose_cfg: dict) -> bool:
    """두 후보가 같은 물체의 인접 부위(손잡이-몸통)인지 판정한다.

    _decide_pose의 부속 손잡이 판정과 같은 기준이다: 작은 쪽 마스크를 크기 비례
    반경으로 팽창시켜 겹침이 body_adjacency_min_px 이상이면 인접으로 본다.
    같은 클래스끼리는 부위 관계가 아니므로 제외한다(머그 두 개가 붙어 있는 경우 등).
    """

    if a.class_id == b.class_id:
        return False
    small, large = (a, b) if a.area_px <= b.area_px else (b, a)
    radius = max(5, int(round(0.15 * math.sqrt(max(small.area_px, 1)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(small.mask.astype(np.uint8), kernel).astype(bool)
    return int((dilated & large.mask).sum()) >= int(pose_cfg.get("body_adjacency_min_px", 20))


def decide_grasp(
    candidates: list[GraspCandidate],
    functional_mask: np.ndarray,
    hand: HandState,
    config: dict,
) -> Decision:
    """후보를 채점해 최적 후보·자세 또는 ALIGN/NO_TARGET을 결정한다."""

    weights = config["weights"]
    norm = config["normalization"]
    pose_cfg = dict(config["pose"])
    # 손 추적이 실측 폭 기반 스케일을 제공하면 설정의 mm_per_px보다 우선한다.
    if getattr(hand, "mm_per_px", None):
        pose_cfg["mm_per_px"] = float(hand.mm_per_px)
    decision_cfg = config["decision"]
    prior = config["human_prior"]

    # 기능 영역까지의 거리 지도 (기능 영역이 없으면 전부 안전).
    if functional_mask.any():
        distance_to_functional = cv2.distanceTransform(
            (~functional_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
    else:
        distance_to_functional = None

    # 1차: 손 적응형 파지점 결정과 거부 규칙 적용.
    # 파지점은 "충분히 안쪽(distance transform ≥ 0.6·최대)"인 픽셀 중 손에서
    # 가장 가까운 점으로 정한다 — 큰 몸통에서 접근 방향 쪽을 잡게 된다.
    # 거리 점수는 최근접 후보 대비 비율이라 전체 거리를 먼저 알아야 한다.
    body_masks = [c.mask for c in candidates if c.class_id == 1]

    eligible: list[tuple[GraspCandidate, float, float, str, float]] = []
    for candidate in candidates:
        distance_map = cv2.distanceTransform(candidate.mask.astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(candidate.mask)
        interior_values = distance_map[ys, xs]
        interior = interior_values >= 0.6 * float(interior_values.max())
        hand_distances = np.hypot(xs - hand.position[0], ys - hand.position[1])
        pick = int(np.flatnonzero(interior)[np.argmin(hand_distances[interior])])
        candidate.grasp_point = (int(xs[pick]), int(ys[pick]))
        mask_distance = float(hand_distances.min())

        x, y = candidate.grasp_point
        if distance_to_functional is not None:
            functional_distance = float(distance_to_functional[y, x])
        else:
            functional_distance = float(norm["safety_radius_px"])
        width_fitness, too_wide = _width_fitness(candidate, pose_cfg)
        pose = _decide_pose(candidate, body_masks, functional_mask, pose_cfg)
        # 유령 후보(가림 유지 스냅샷)는 스냅샷 시점에 같은 검사를 통과했고, 가림 중의
        # 기능부 오검출·스케일 오염이 재검사를 오염시킬 수 있으므로 거부 검사를 면제한다.
        is_ghost = bool(getattr(candidate, "_ghost", False))
        if too_wide and not is_ghost:
            candidate.scores = {"rejected": 1.0}
            candidate.scores["reject_reason"] = "max_grasp_width_exceeded"  # type: ignore[assignment]
            continue
        if (functional_distance < float(decision_cfg["functional_hard_floor_px"])
                and not is_ghost):
            candidate.scores = {"rejected": 1.0}
            candidate.scores["reject_reason"] = "too_close_to_functional"  # type: ignore[assignment]
            continue
        eligible.append((candidate, mask_distance, width_fitness, pose, functional_distance))

    scored: list[GraspCandidate] = []
    nearest = min((entry[1] for entry in eligible), default=0.0)
    for candidate, distance, width_fitness, pose, functional_distance in eligible:
        x, y = candidate.grasp_point
        distance_score = float(nearest / distance) if distance > 1e-6 else 1.0
        # 방향 정렬은 파지점까지의 변위 벡터로 계산한다 (거리 점수와 별개).
        point_distance = math.hypot(x - hand.position[0], y - hand.position[1])
        if hand.direction is not None and point_distance > 1e-6:
            unit = ((x - hand.position[0]) / point_distance, (y - hand.position[1]) / point_distance)
            cosine = unit[0] * hand.direction[0] + unit[1] * hand.direction[1]
            approach_score = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        else:
            approach_score = 0.5  # 방향 정보가 없으면 중립
        safety_score = float(np.clip(functional_distance / float(norm["safety_radius_px"]), 0.0, 1.0))
        human_score = float(prior.get(candidate.class_name, 0.0))

        components = {
            "confidence": candidate.confidence,
            "distance": distance_score,
            "approach": approach_score,
            "width_fit": width_fitness,
            "functional_safety": safety_score,
            "human_prior": human_score,
        }
        total = sum(weights[key] * value for key, value in components.items())
        candidate.scores = {**components, "total": total, "pose": pose}  # type: ignore[dict-item]
        scored.append(candidate)

    if not scored:
        return Decision(state="NO_TARGET", reason="유효한 파지 후보가 없습니다")

    ranked = sorted(scored, key=lambda c: c.scores["total"], reverse=True)
    top = ranked[0]

    # 인접 부위 타이브레이크: 1·2위가 같은 물체의 손잡이-몸통이면 점수 대신
    # 손이 향하는 쪽(파지점이 손에 가까운 쪽)을 승자로 확정한다. 인접 부위
    # 사이에서는 거리·방향 항이 포화되어 점수식이 검출 신뢰도 잡음으로 갈리기
    # 때문이다 (2026-08-25 파란 머그 POWER/WRAP 요동 분석). 서로 다른 물체 간
    # 선택은 아래 기존 점수·margin 규칙을 그대로 따른다.
    if (
        len(ranked) > 1
        and bool(decision_cfg.get("adjacent_tiebreak", True))
        and _parts_adjacent(ranked[0], ranked[1], pose_cfg)
    ):
        winner = min(ranked[:2], key=lambda c: math.hypot(
            c.grasp_point[0] - hand.position[0], c.grasp_point[1] - hand.position[1]))
        winner.scores["adjacent_tiebreak"] = 1.0
        if winner is not top:
            ranked.remove(winner)
            ranked.insert(0, winner)
            top = winner
        top_score = float(top.scores["total"])
        if top_score < float(decision_cfg["min_score"]):
            return Decision(state="ALIGN", ranked=ranked,
                            reason=f"최고 점수 {top_score:.2f} < {decision_cfg['min_score']}")
        return Decision(state="GRASP", candidate=top, pose=str(top.scores["pose"]),
                        ranked=ranked, reason="인접 부위 타이브레이크: 손에 가까운 부위 확정")

    top_score = float(top.scores["total"])
    if top_score < float(decision_cfg["min_score"]):
        return Decision(state="ALIGN", ranked=ranked,
                        reason=f"최고 점수 {top_score:.2f} < {decision_cfg['min_score']}")
    if len(ranked) > 1:
        margin = top_score - float(ranked[1].scores["total"])
        if margin < float(decision_cfg["score_margin"]):
            return Decision(state="ALIGN", ranked=ranked,
                            reason=f"1·2위 점수 차 {margin:.2f} < {decision_cfg['score_margin']}")
    return Decision(state="GRASP", candidate=top, pose=str(top.scores["pose"]),
                    ranked=ranked, reason="최적 후보 확정")
