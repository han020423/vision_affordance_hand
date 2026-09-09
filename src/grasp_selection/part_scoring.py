"""부위 수준 파지 결정 — 파지점 없는 변형 (run_realtime_rfdetr_part.py 전용).

2026-08-26 사용자 제안: 파지점을 정하지 않고 **손 접근과 물체 부위**만으로 파지
자세를 결정한다. 근거: 이 시스템의 공유 제어 구조에서 조준은 사람이 하고 구동은
접촉(스톨) 적응이 담당하므로, 픽셀 파지점은 구동에 쓰이지 않으면서 가림에 가장
취약한 산출물이었다. 반자율 의수 문헌(Došen 2010, Marković 2014)도 비전은 파지
유형·개구를 결정하고 접촉점은 정하지 않는다.

기존 scoring.decide_grasp와 실시간 스크립트는 그대로 보존한다(A/B 비교용).
자세 구조 규칙(_decide_pose)·인접 판정(_parts_adjacent)·폭 검사(_width_fitness)는
scoring의 것을 그대로 공유해 두 경로의 규칙이 갈라지지 않게 한다.

점 기반 결정과의 차이:
- 접근 방향 정렬: 파지점 대신 "부위 마스크에서 손과 가장 가까운 픽셀"로의 변위 사용
- 기능부 회피: "파지점이 기능부와 12px 이내" 대신 **부위 얽힘 비율** — 부위 픽셀 중
  기능부 하한 거리 안에 있는 비율이 functional_part_fraction_max(기본 0.5)를 넘으면
  거부. 도구 자루처럼 기능부와 맞닿아도 대부분이 안전한 부위는 통과하고,
  날을 따라 붙은 조각처럼 얽힌 부위만 거부된다.
- 인접 부위 타이브레이크: 손-파지점 거리 대신 손-부위 마스크 최근접 거리
- candidate.grasp_point는 계산·표시용으로 남지만 결정에는 쓰지 않는다
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .candidates import GraspCandidate
from .scoring import Decision, HandState, _decide_pose, _parts_adjacent, _width_fitness

__all__ = ["decide_grasp_by_part"]


def decide_grasp_by_part(
    candidates: list[GraspCandidate],
    functional_mask: np.ndarray,
    hand: HandState,
    config: dict,
) -> Decision:
    """부위 수준으로 후보를 채점해 최적 부위·자세 또는 ALIGN/NO_TARGET을 결정한다."""

    weights = config["weights"]
    norm = config["normalization"]
    pose_cfg = dict(config["pose"])
    if getattr(hand, "mm_per_px", None):
        pose_cfg["mm_per_px"] = float(hand.mm_per_px)
    decision_cfg = config["decision"]
    prior = config["human_prior"]
    hard_floor = float(decision_cfg["functional_hard_floor_px"])
    entangle_max = float(decision_cfg.get("functional_part_fraction_max", 0.5))

    if functional_mask.any():
        distance_to_functional = cv2.distanceTransform(
            (~functional_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
    else:
        distance_to_functional = None

    body_masks = [c.mask for c in candidates if c.class_id == 1]

    eligible = []
    for candidate in candidates:
        ys, xs = np.nonzero(candidate.mask)
        if len(xs) == 0:
            continue
        hand_distances = np.hypot(xs - hand.position[0], ys - hand.position[1])
        pick = int(np.argmin(hand_distances))
        near_point = (float(xs[pick]), float(ys[pick]))   # 방향 계산용 내부값 (파지점 아님)
        mask_distance = float(hand_distances[pick])

        is_ghost = bool(getattr(candidate, "_ghost", False))
        if distance_to_functional is not None:
            part_distances = distance_to_functional[ys, xs]
            entangled = float((part_distances < hard_floor).mean())
            functional_distance = float(part_distances.mean())
        else:
            entangled = 0.0
            functional_distance = float(norm["safety_radius_px"])

        width_fitness, too_wide = _width_fitness(candidate, pose_cfg)
        pose = _decide_pose(candidate, body_masks, functional_mask, pose_cfg)
        # 유령 후보는 스냅샷 시점에 검증을 통과했으므로 거부 검사 면제 (scoring과 동일 원칙)
        if too_wide and not is_ghost:
            candidate.scores = {"rejected": 1.0}
            candidate.scores["reject_reason"] = "max_grasp_width_exceeded"  # type: ignore[assignment]
            continue
        if entangled > entangle_max and not is_ghost:
            candidate.scores = {"rejected": 1.0}
            candidate.scores["reject_reason"] = "functional_entangled"  # type: ignore[assignment]
            continue
        eligible.append((candidate, mask_distance, near_point, width_fitness, pose,
                         functional_distance))

    scored = []
    nearest = min((entry[1] for entry in eligible), default=0.0)
    for candidate, mask_distance, near_point, width_fitness, pose, functional_distance in eligible:
        distance_score = float(nearest / mask_distance) if mask_distance > 1e-6 else 1.0
        displacement = math.hypot(near_point[0] - hand.position[0],
                                  near_point[1] - hand.position[1])
        if hand.direction is not None and displacement > 1e-6:
            unit = ((near_point[0] - hand.position[0]) / displacement,
                    (near_point[1] - hand.position[1]) / displacement)
            cosine = unit[0] * hand.direction[0] + unit[1] * hand.direction[1]
            approach_score = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        else:
            approach_score = 0.5
        safety_score = float(np.clip(functional_distance / float(norm["safety_radius_px"]),
                                     0.0, 1.0))
        components = {
            "confidence": candidate.confidence,
            "distance": distance_score,
            "approach": approach_score,
            "width_fit": width_fitness,
            "functional_safety": safety_score,
            "human_prior": float(prior.get(candidate.class_name, 0.0)),
        }
        total = sum(weights[key] * value for key, value in components.items())
        if bool(getattr(candidate, "_ghost", False)):
            # 유령(기억) 후보는 점수를 할인해, 실측 후보가 나타나면 즉시 양보하고
            # 혼자 남았을 때만 1위가 되게 한다. 할인이 없으면 목표를 바꿀 때 이전
            # 물체의 유령이 순위·margin 게이트를 점유해 전환이 굼떠진다
            # (2026-08-26 실물 체감 보고).
            total *= float(decision_cfg.get("ghost_score_factor", 0.85))
        candidate.scores = {**components, "total": total, "pose": pose,  # type: ignore[dict-item]
                            "mask_distance": mask_distance}
        scored.append(candidate)

    if not scored:
        return Decision(state="NO_TARGET", reason="유효한 파지 부위가 없습니다")

    ranked = sorted(scored, key=lambda c: c.scores["total"], reverse=True)
    top = ranked[0]

    # 인접 부위 타이브레이크 (scoring과 같은 규칙, 거리만 부위 마스크 기준)
    if (
        len(ranked) > 1
        and bool(decision_cfg.get("adjacent_tiebreak", True))
        and _parts_adjacent(ranked[0], ranked[1], pose_cfg)
    ):
        winner = min(ranked[:2], key=lambda c: c.scores["mask_distance"])
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
                    ranked=ranked, reason="최적 부위 확정")
