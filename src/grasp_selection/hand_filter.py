"""로봇손 오검출 억제 필터.

분할 모델이 로봇손을 파지 후보로 오인하는 문제(2026-08-24 젯슨 실물 시험에서 확인)를
후보 단계에서 걸러낸다. 두 가지 신호를 쓴다.

1. **시간 안정성**: 이 시스템은 카메라와 물체가 고정이고 움직이는 것은 로봇손뿐이다.
   따라서 몇 프레임 연속 같은 자리에 있는 후보만 실제 물체로 인정한다. 화면을
   지나가는 손이 만드는 오검출은 위치가 계속 변하므로 자동으로 걸러진다.
   (`PROJECT_CONTEXT.md` 10절의 "안정적으로 추적되지 않은 후보 거부" 규칙의 구현)
2. **손 영역 겹침 거부**(선택): 손 추적기가 아는 손 박스 안에 후보 면적의 큰 비율이
   들어 있으면 손으로 보고 거부한다. 접근 중 목표물이 손과 살짝 겹치는 경우를
   지키기 위해 "면적 비율" 기준을 쓰고, 추적이 유효한 프레임에만 적용해야 한다
   (오래된 박스로 거부하면 진짜 목표를 죽인다 — 호출자가 hand_box=None으로 끈다).
3. **파지점 안정화**(stabilize_points): 같은 정지 가정에서, 손이 물체를 가려
   마스크가 잘리면 잘린 마스크로 새로 계산한 파지점은 가림 경계로 끌려간다.
   물체는 움직이지 않았으므로 가림 동안에는 마지막 안정 파지점을 유지하고,
   마스크가 온전할 때는 EMA로 떨림만 제거한다 (2026-08-25 O 실물 시험에서
   손 접근 시 파지점 소실·요동 확인 → 보강).
4. **클래스 다수결**: 트랙 매칭은 클래스 무관(위치·면적)으로 하고, 트랙이 최근
   관측 창의 다수결로 클래스를 확정해 후보에 덮어쓴다. 결정 경계에 걸린 물체
   (종이컵 등)가 handle↔body로 프레임마다 뒤집히면 클래스별 트랙은 매번 리셋되어
   후보가 영원히 unstable로 거부되던 문제의 해결이다 (2026-08-25 실물 확인).
5. **유령 후보**: 검증되던 트랙의 후보가 사라지면(근접 가림: 모델 미검출·손 픽셀
   통제거·손 박스 거부) 마지막 검증 스냅샷을 ghost_ttl_frames 동안 대신 내보낸다.
   파지점 안정화(3)는 후보가 살아 있어야 작동하므로, 후보 자체가 소멸하는 최종
   접근 구간은 이 장치가 맡는다 (2026-08-26 사용자 보고 대응).

근본 해결(손 포함 음성 데이터로 미세조정)은 별도 작업이며, 이 필터는 그 전까지의
방어이자 이후에도 남는 안전망이다.
"""

from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .candidates import MODEL_CLASS_NAMES, GraspCandidate


@dataclass
class _Track:
    """정지 물체 후보의 프레임 간 궤적."""

    class_id: int                  # 다수결로 확정된 클래스 (관측 클래스가 아님)
    point: tuple[float, float]     # 후보 파지점(distance transform 최대점)의 EMA
    area: float                    # 면적의 EMA
    age: int = 1                   # 연속 관측 프레임 수
    last_seen: int = 0             # 마지막으로 관측된 프레임 번호
    full_area: float = 0.0         # 비가림 상태 면적 추정 (천천히 감쇠하는 최댓값)
    stable_point: tuple[float, float] | None = None  # 검증 통과 파지점의 EMA (가림 시 유지값)
    class_history: deque = field(default_factory=deque, repr=False)  # 최근 관측 클래스 창
    last_candidate: GraspCandidate | None = field(default=None, repr=False)  # 검증 통과 스냅샷(유령 후보 재료)
    ghost_age: int = 0             # 유령 TTL 카운터 (손이 덮고 있는 동안은 세지 않음)


@dataclass
class HandSuppressionFilter:
    """시간 안정성 + 손 겹침 기반 후보 필터. 프레임마다 update()를 호출한다."""

    min_stable_frames: int = 3     # 이 프레임 수 이상 같은 자리에 있어야 채점 대상
    match_radius_px: int = 60      # 같은 트랙으로 볼 파지점 이동 허용치
    area_ratio_max: float = 2.5    # 같은 트랙으로 볼 면적 변화 허용 배율
    hand_overlap_max: float = 0.5  # 후보 면적 중 손 박스 안 비율이 이 이상이면 거부
    hand_box_margin: float = 0.15  # 손 박스를 각 변 크기의 이 비율만큼 넓혀서 판정
    miss_ttl_frames: int = 10      # 이 프레임 수 동안 안 보이면 트랙 폐기
    point_ema_alpha: float = 0.4   # 파지점 안정화 EMA에서 새 관측의 가중치
    occlusion_hold_ratio: float = 0.7  # 면적이 full_area의 이 비율 미만이면 가림으로 보고 점 유지
    full_area_decay: float = 0.98  # full_area의 프레임당 감쇠 (물체 크기 변화에 서서히 적응)
    class_vote_window: int = 7     # 클래스 다수결에 쓰는 최근 관측 창 크기
    class_switch_min: int = 5      # 창 안에서 다른 클래스가 이 횟수 이상이어야 확정 클래스 전환
    ghost_ttl_frames: int = 20     # 유령 후보 유지 프레임 수. 단, 손 박스가 트랙을 덮고 있는
                                   # 동안(가림 설명 성립)은 세지 않는다 — 손이 떠난 뒤 ~2.4초
    ghost_hard_cap_frames: int = 200  # 어떤 경우에도 이 프레임(~23초)을 넘겨 유지하지 않는 안전 상한
    ghost_replace_overlap: float = 0.3  # 스냅샷 면적의 이 비율 이상을 '다른' 안정 후보가 덮으면
                                        # 교체로 보고 기억을 즉시 폐기 (물체 교체 잔상 방지)

    _tracks: list[_Track] = field(default_factory=list, repr=False)
    _frame: int = field(default=0, repr=False)
    _kept_map: dict[int, _Track] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, selection_config: dict) -> "HandSuppressionFilter":
        """grasp_selection.yaml의 suppression 절로 생성한다(없으면 기본값)."""

        section = dict(selection_config.get("suppression", {}) or {})
        return cls(
            min_stable_frames=int(section.get("min_stable_frames", 3)),
            match_radius_px=int(section.get("match_radius_px", 60)),
            area_ratio_max=float(section.get("area_ratio_max", 2.5)),
            hand_overlap_max=float(section.get("hand_overlap_max", 0.5)),
            hand_box_margin=float(section.get("hand_box_margin", 0.15)),
            miss_ttl_frames=int(section.get("miss_ttl_frames", 10)),
            point_ema_alpha=float(section.get("point_ema_alpha", 0.4)),
            occlusion_hold_ratio=float(section.get("occlusion_hold_ratio", 0.7)),
            full_area_decay=float(section.get("full_area_decay", 0.98)),
            class_vote_window=int(section.get("class_vote_window", 7)),
            class_switch_min=int(section.get("class_switch_min", 5)),
            ghost_ttl_frames=int(section.get("ghost_ttl_frames", 20)),
            ghost_hard_cap_frames=int(section.get("ghost_hard_cap_frames", 200)),
            ghost_replace_overlap=float(section.get("ghost_replace_overlap", 0.3)),
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0
        self._kept_map.clear()

    # ------------------------------------------------------------------ 내부

    def _hand_overlap_fraction(self, candidate: GraspCandidate, hand_box) -> float:
        """후보 마스크 면적 중 (넓힌) 손 박스 안에 든 비율."""

        x1, y1, x2, y2 = hand_box
        margin_x = int(round((x2 - x1) * self.hand_box_margin))
        margin_y = int(round((y2 - y1) * self.hand_box_margin))
        height, width = candidate.mask.shape
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        if x2 <= x1 or y2 <= y1 or candidate.area_px <= 0:
            return 0.0
        inside = int(candidate.mask[y1:y2, x1:x2].sum())
        return inside / candidate.area_px

    def _point_in_expanded_box(self, point: tuple[float, float],
                               box: tuple[int, int, int, int]) -> bool:
        """점이 (margin만큼 넓힌) 손 박스 안에 있는지 판정한다 (유령 TTL 동결용)."""

        x1, y1, x2, y2 = box
        margin_x = (x2 - x1) * self.hand_box_margin
        margin_y = (y2 - y1) * self.hand_box_margin
        return (x1 - margin_x <= point[0] <= x2 + margin_x
                and y1 - margin_y <= point[1] <= y2 + margin_y)

    @staticmethod
    def _point_inside(point: tuple[float, float], mask: np.ndarray) -> bool:
        """트랙 파지점이 후보 마스크 안에 있는지 O(1)로 판정한다."""

        x, y = int(round(point[0])), int(round(point[1]))
        height, width = mask.shape
        return 0 <= x < width and 0 <= y < height and bool(mask[y, x])

    def _match_track(self, candidate: GraspCandidate,
                     claimed: set[int] | None = None) -> _Track | None:
        """후보와 이어지는 트랙을 찾는다. 매칭은 클래스 무관(위치·면적 기반)이다.

        클래스를 매칭 조건에 넣으면 결정 경계에 걸린 물체(종이컵 등)가 handle↔body로
        뒤집힐 때마다 트랙이 리셋되어 영원히 unstable로 거부된다(2026-08-25 확인).
        클래스는 매칭이 아니라 트랙의 다수결(update 참조)로 다룬다. claimed는 이번
        프레임에 이미 다른 후보가 가져간 트랙 — 인접한 손잡이·몸통 후보가 한 트랙에
        합쳐지지 않게 제외한다.
        """

        point = candidate.grasp_point
        best = None
        best_distance = None
        for track in self._tracks:
            if claimed is not None and id(track) in claimed:
                continue
            distance = float(np.hypot(point[0] - track.point[0], point[1] - track.point[1]))
            # 손이 물체를 가리면 잘린 마스크의 파지점이 크게 튀고 면적이 급감해
            # 거리·면적 기준만으로는 트랙이 끊긴다(→ unstable → 접근 중 점 소실).
            # 정지 물체 가정에서 트랙의 옛 파지점이 여전히 후보 마스크 안에 있으면
            # 같은 물체의 보이는 부분으로 보고 매칭을 허용한다. 단 비가림 면적
            # (full_area) 대비 과도 성장(오검출 병합)은 여전히 차단한다.
            contains = self._point_inside(track.point, candidate.mask)
            if distance > self.match_radius_px and not contains:
                continue
            ratio = candidate.area_px / max(track.area, 1.0)
            if not (1.0 / self.area_ratio_max <= ratio <= self.area_ratio_max):
                if not (contains and candidate.area_px <= self.area_ratio_max * track.full_area):
                    continue
            if best is None or distance < best_distance:
                best, best_distance = track, distance
        return best

    # ------------------------------------------------------------------ 공개 API

    def update(
        self,
        candidates: list[GraspCandidate],
        hand_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[GraspCandidate], dict[str, int]]:
        """이번 프레임 후보를 필터링한다.

        hand_box는 손 추적이 이번 프레임에 유효할 때만 넘겨야 한다(스테일 박스 금지).
        반환: (통과한 후보, {"unstable": n, "hand_overlap": n}).
        """

        self._frame += 1
        self._kept_map = {}
        kept: list[GraspCandidate] = []
        suppressed = {"unstable": 0, "hand_overlap": 0}
        claimed: set[int] = set()   # 이번 프레임에 이미 후보가 배정된 트랙

        for candidate in candidates:
            if hand_box is not None and (
                self._hand_overlap_fraction(candidate, hand_box) >= self.hand_overlap_max
            ):
                suppressed["hand_overlap"] += 1
                continue   # 손으로 판정: 트랙도 만들지 않는다

            track = self._match_track(candidate, claimed)
            if track is None:
                track = _Track(
                    class_id=candidate.class_id,
                    point=(float(candidate.grasp_point[0]), float(candidate.grasp_point[1])),
                    area=float(candidate.area_px),
                    age=1,
                    last_seen=self._frame,
                    full_area=float(candidate.area_px),
                    class_history=deque([candidate.class_id], maxlen=self.class_vote_window),
                )
                self._tracks.append(track)
            else:
                # 프레임을 건너뛰었어도 TTL 안이면 이어 본다(검출 깜빡임 허용)
                track.age += 1
                track.point = (
                    0.5 * track.point[0] + 0.5 * candidate.grasp_point[0],
                    0.5 * track.point[1] + 0.5 * candidate.grasp_point[1],
                )
                track.area = 0.5 * track.area + 0.5 * candidate.area_px
                track.last_seen = self._frame
                track.ghost_age = 0
                # 가림으로 잠깐 줄어든 면적에 끌려가지 않도록 천천히 감쇠하는 최댓값을 유지한다.
                track.full_area = max(float(candidate.area_px), track.full_area * self.full_area_decay)
                # 클래스 다수결: 결정 경계에 걸린 물체(종이컵 등)의 handle↔body
                # 뒤집힘을 억제한다. 한두 프레임의 반대 관측으로는 안 바뀌고,
                # 창 안에서 class_switch_min 이상 관측된 클래스로만 전환한다.
                track.class_history.append(candidate.class_id)
                counts = Counter(track.class_history)
                challenger = max(
                    (item for item in counts.items() if item[0] != track.class_id),
                    key=lambda item: item[1], default=None,
                )
                if challenger is not None and challenger[1] >= self.class_switch_min:
                    track.class_id = challenger[0]
            claimed.add(id(track))

            # 후보 클래스를 트랙의 확정 클래스로 덮어쓴다 — 이후 단계(자세 결정,
            # 사전확률, 타이브레이크)가 프레임 잡음이 아닌 다수결 클래스를 쓰게 한다.
            if candidate.class_id != track.class_id:
                candidate.class_id = track.class_id
                candidate.class_name = MODEL_CLASS_NAMES.get(
                    track.class_id, f"class_{track.class_id}"
                )

            if track.age >= self.min_stable_frames:
                kept.append(candidate)
                self._kept_map[id(candidate)] = track
            else:
                suppressed["unstable"] += 1

        # 유령 후보: 방금까지 검증되던 트랙의 후보가 이번 프레임에 사라졌으면
        # (근접 가림의 세 증상 — 모델 미검출, 손 픽셀 통제거, 손 박스 거부)
        # 마지막 검증 스냅샷을 짧은 TTL 동안 대신 내보낸다. 정지 장면 가정에서
        # 물체는 손 뒤 그 자리에 그대로 있으므로, 접근 마지막 구간에서 파지점이
        # 소멸하는 문제(2026-08-26 사용자 보고)의 해결이다. 유령은 트랙 상태를
        # 갱신하지 않으므로 물체가 실제로 치워지면 TTL 뒤에 자연 소멸한다.
        suppressed["ghost"] = 0
        # 교체 증거 수집: 이번 프레임에 확정(안정)된 실측 후보들. 이들이 어떤 트랙의
        # 스냅샷 자리를 덮고 있는데 그 트랙과 매칭되지 않았다면, 옛 물체는 다른
        # 물체로 교체된 것이므로 기억을 폐기한다 (교체 후 잔상 방지, 2026-08-26).
        # 가림 파편은 안정(kept)에 도달하기 어려워 이 규칙에 걸리지 않는다.
        stable_real = [(candidate, self._kept_map.get(id(candidate))) for candidate in kept
                       if not getattr(candidate, "_ghost", False)]
        for track in self._tracks:
            if track.last_candidate is None or track.age < self.min_stable_frames:
                continue
            missed = self._frame - track.last_seen
            if missed < 1:
                continue   # 이번 프레임에 실측 후보가 있음 → 유령 불필요
            snapshot_mask = track.last_candidate.mask
            snapshot_area = max(int(track.last_candidate.area_px), 1)
            replaced = any(
                other_track is not track
                and int((candidate.mask & snapshot_mask).sum()) / snapshot_area
                >= self.ghost_replace_overlap
                for candidate, other_track in stable_real
            )
            if replaced:
                track.last_candidate = None   # 기억 폐기 → 이후 유령 없음
                continue
            # 손 박스가 트랙(옛 파지점) 위를 덮고 있는 동안은 "가림" 설명이 성립하므로
            # TTL을 세지 않는다 — 호버·정렬이 아무리 길어져도 유지된다. 손이 떠난
            # 뒤부터 TTL을 세고, 어떤 경우에도 하드 캡은 넘기지 않는다(물체가 실제로
            # 치워진 경우의 유령 잔상 안전장치).
            covered = hand_box is not None and self._point_in_expanded_box(track.point, hand_box)
            if not covered:
                track.ghost_age += 1
            if track.ghost_age > self.ghost_ttl_frames or missed > self.ghost_hard_cap_frames:
                continue
            ghost = copy.copy(track.last_candidate)
            ghost.scores = {}
            ghost._ghost = True   # type: ignore[attr-defined]  # stabilize_points가 HOLD 표시에 사용
            kept.append(ghost)
            self._kept_map[id(ghost)] = track
            suppressed["ghost"] += 1

        # 트랙 폐기 시한: 검증 스냅샷이 있는 트랙은 유령 하드 캡까지 살리고,
        # 검증된 적 없는 트랙은 기존 TTL로 버린다.
        def keep_track(track: _Track) -> bool:
            ttl = (self.ghost_hard_cap_frames if track.last_candidate is not None
                   else self.miss_ttl_frames)
            return self._frame - track.last_seen <= max(ttl, self.miss_ttl_frames)

        self._tracks = [track for track in self._tracks if keep_track(track)]
        return kept, suppressed

    def stabilize_points(self, candidates: list[GraspCandidate]) -> int:
        """검증을 통과한 후보들의 파지점을 트랙 기억으로 안정화한다.

        decide_grasp가 손 적응 파지점을 확정한 **뒤에**, 그 프레임의 ranked 후보로
        호출해야 한다(거부된 후보의 점은 안정값에 반영하지 않기 위해서다).
        고정 카메라·정지 물체 가정(모듈 docstring 1번 신호)에 따라:

        - 마스크가 온전할 때(면적 ≥ occlusion_hold_ratio·full_area): 안정 파지점을
          EMA로 갱신하고 그 값을 후보에 되쓴다 — 분할 잡음의 프레임 간 떨림 제거.
          EMA 결과가 마스크 밖이면(오목한 손잡이에서 두 내부점의 중간이 밖일 수
          있음) 새 관측값으로 재초기화한다.
        - 가림으로 면적이 줄었을 때: 잘린 마스크에서 새로 계산한 점은 가림 경계로
          끌려가므로 버리고 마지막 안정 파지점을 그대로 유지한다(물체는 움직이지
          않았다). 유지된 점은 손 뒤에 가려져 있어도 물리적으로 유효한 위치다.

        유지된 후보는 scores["point_held"]=1.0 으로 표시한다. 반환: 유지된 후보 수.
        """

        held = 0
        for candidate in candidates:
            track = self._kept_map.get(id(candidate))
            if track is None:
                continue
            if getattr(candidate, "_ghost", False):
                # 유령 후보: 실측 마스크가 아니므로 안정값 갱신 재료로 쓰지 않고
                # 유지 표시만 남긴다 (파지점은 스냅샷 마스크 기준으로 이미 계산됨).
                candidate.scores["point_held"] = 1.0
                held += 1
                continue
            # 검증(순위 도달)을 통과한 실측 후보만 유령 후보의 재료로 저장한다.
            snapshot = copy.copy(candidate)
            snapshot.scores = dict(candidate.scores)
            track.last_candidate = snapshot
            new_point = (float(candidate.grasp_point[0]), float(candidate.grasp_point[1]))
            occluded = candidate.area_px < self.occlusion_hold_ratio * track.full_area
            if occluded and track.stable_point is not None:
                candidate.grasp_point = (
                    int(round(track.stable_point[0])), int(round(track.stable_point[1]))
                )
                candidate.scores["point_held"] = 1.0
                held += 1
                continue
            if track.stable_point is None:
                track.stable_point = new_point
            else:
                alpha = self.point_ema_alpha
                smoothed = (
                    (1.0 - alpha) * track.stable_point[0] + alpha * new_point[0],
                    (1.0 - alpha) * track.stable_point[1] + alpha * new_point[1],
                )
                track.stable_point = (
                    smoothed if self._point_inside(smoothed, candidate.mask) else new_point
                )
            candidate.grasp_point = (
                int(round(track.stable_point[0])), int(round(track.stable_point[1]))
            )
        return held
