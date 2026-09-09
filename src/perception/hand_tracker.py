"""로봇손 비전 추적: 검출기(플러그인) + 이동 방향 추정 + 스케일 추정.

ArUco 마커 없이 로봇손(Brunel Hand) 자체를 검출해 후보 선택의 HandState를
만든다. 검출 백엔드는 교체 가능하다.

- yolo_world: YOLO-World 제로샷("robot hand" 등 프롬프트), 학습 불필요 (1단계)
- custom: 촬영 데이터로 학습한 전용 검출기 (2단계)

접근 방향은 마커 자세 대신 손 위치 변화의 지수이동평균(속도 벡터)으로 추정한다.
실측 손 폭(hand_width_mm)이 설정되면 화면 폭과의 비로 mm_per_px를 추정해
폭 기준(집기/감아쥐기, 최대 개구)을 mm 단위로 쓸 수 있게 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.grasp_selection.scoring import HandState


@dataclass
class HandTrackerConfig:
    """손 추적 설정 (configs/hardware/hand_tracker.yaml)."""

    backend: str = "yolo_world"
    weights: str = "models/pretrained/yolov8s-worldv2.pt"
    prompts: list[str] = field(default_factory=lambda: ["robot hand"])
    custom_weights: str | None = None
    conf_threshold: float = 0.2
    imgsz: int = 640
    device: str = "cpu"
    detect_every: int = 4
    max_missing_frames: int = 30
    min_box_px: int = 40
    direction_ema_alpha: float = 0.3
    min_speed_px: float = 4.0
    hand_width_mm: float | None = None
    width_px_decay: float = 0.995            # mm/px 기준 손 폭(감쇠 최댓값)의 프레임당 감쇠
    # motion 백엔드(배경 차분) 전용 파라미터
    motion_history: int = 300
    motion_var_threshold: float = 25.0       # 낮을수록 민감 (어두운 손·어두운 배경 대비)
    motion_learning_rate: float = 0.01       # 높을수록 잔상(ghost)이 빨리 사라짐
    motion_min_area_px: int = 8000
    motion_blur: int = 5
    motion_morph_kernel: int = 9
    motion_position_mode: str = "leading"   # leading(이동 방향 선단) | centroid
    motion_warmup_frames: int = 30           # 배경 모델이 안정되기 전에는 검출하지 않음
    motion_use_framediff: bool = True        # 프레임 간 차분으로 "지금 움직이는" 덩어리만 인정
    motion_framediff_threshold: int = 12
    motion_framediff_dilate: int = 25


def load_hand_tracker_config(path: Path) -> HandTrackerConfig:
    """YAML을 읽어 HandTrackerConfig로 해석한다."""

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return HandTrackerConfig(
        backend=str(document.get("backend", "yolo_world")),
        weights=str(document.get("weights", "models/pretrained/yolov8s-worldv2.pt")),
        prompts=list(document.get("prompts", ["robot hand"])),
        custom_weights=document.get("custom_weights"),
        conf_threshold=float(document.get("conf_threshold", 0.2)),
        imgsz=int(document.get("imgsz", 640)),
        device=str(document.get("device", "cpu")),
        detect_every=int(document.get("detect_every", 4)),
        max_missing_frames=int(document.get("max_missing_frames", 30)),
        min_box_px=int(document.get("min_box_px", 40)),
        direction_ema_alpha=float(document.get("direction_ema_alpha", 0.3)),
        min_speed_px=float(document.get("min_speed_px", 4.0)),
        hand_width_mm=document.get("hand_width_mm"),
        width_px_decay=float(document.get("width_px_decay", 0.995)),
        motion_history=int(document.get("motion_history", 300)),
        motion_var_threshold=float(document.get("motion_var_threshold", 40.0)),
        motion_learning_rate=float(document.get("motion_learning_rate", 0.002)),
        motion_min_area_px=int(document.get("motion_min_area_px", 4000)),
        motion_blur=int(document.get("motion_blur", 5)),
        motion_morph_kernel=int(document.get("motion_morph_kernel", 9)),
        motion_position_mode=str(document.get("motion_position_mode", "leading")),
        motion_warmup_frames=int(document.get("motion_warmup_frames", 30)),
        motion_use_framediff=bool(document.get("motion_use_framediff", True)),
        motion_framediff_threshold=int(document.get("motion_framediff_threshold", 12)),
        motion_framediff_dilate=int(document.get("motion_framediff_dilate", 25)),
    )


class HandMotionEstimator:
    """검출 위치 시퀀스에서 접근 방향과 스케일을 추정한다 (모델 비의존, 테스트 가능)."""

    def __init__(self, ema_alpha: float, min_speed_px: float, hand_width_mm: float | None,
                 width_px_decay: float = 0.995) -> None:
        self.ema_alpha = ema_alpha
        self.min_speed_px = min_speed_px
        self.hand_width_mm = hand_width_mm
        self.width_px_decay = width_px_decay
        self.last_position: tuple[float, float] | None = None
        self.velocity: list[float] = [0.0, 0.0]
        self._width_px_ref: float | None = None   # 손 폭 px의 천천히 감쇠하는 최댓값

    def reset(self) -> None:
        """손을 놓쳤을 때 이동 이력을 초기화한다."""

        self.last_position = None
        self.velocity = [0.0, 0.0]
        self._width_px_ref = None

    def observe(self, position: tuple[float, float], width_px: float | None) -> HandState:
        """새 관측 위치로 속도 EMA를 갱신하고 HandState를 만든다."""

        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            a = self.ema_alpha
            self.velocity[0] = (1 - a) * self.velocity[0] + a * dx
            self.velocity[1] = (1 - a) * self.velocity[1] + a * dy
        self.last_position = position
        speed = math.hypot(self.velocity[0], self.velocity[1])
        direction = None
        if speed >= self.min_speed_px:
            direction = (self.velocity[0] / speed, self.velocity[1] / speed)
        # 부분 검출·화면 가장자리 잘림은 손 박스를 실제보다 "작게"만 만든다. 순간값
        # 대신 천천히 감쇠하는 최댓값을 mm/px 기준으로 삼아, 잘린 박스가 스케일을
        # 부풀려 폭 필터가 정상 몸통 후보를 죽이는 연쇄를 막는다
        # (2026-08-25 O 실물 시험: body가 width로 반복 탈락 → POWER 불발의 원인).
        if width_px and width_px > 0:
            if self._width_px_ref is None:
                self._width_px_ref = float(width_px)
            else:
                self._width_px_ref = max(float(width_px), self._width_px_ref * self.width_px_decay)
        mm_per_px = None
        if self.hand_width_mm and self._width_px_ref:
            mm_per_px = float(self.hand_width_mm) / float(self._width_px_ref)
        return HandState(position=position, direction=direction, mm_per_px=mm_per_px)


class SegmentationHandSelector:
    """분할 robot_hand 검출 중 손으로 채택할 하나를 고른다.

    검은 정지 물체가 robot_hand로 오검출되어 손 추적을 납치하는 문제
    (2026-08-25 O 실물 시험)의 실행 시점 방어다.

    - **획득**(추적 없음): 신뢰도가 가장 높은 검출을 즉시 채택한다. 정지한 손도
      바로 잡힌다. (min_motion_frames > 0으로 주면 "프레임 간 min_motion_px 이상
      이동이 연속"인 검출만 획득하는 움직임 게이트가 켜지지만, 정지한 손을 못
      잡는 부작용 때문에 2026-08-25 사용자 결정으로 기본 비활성.)
    - **유지**(추적 중): 직전 채택 중심에서 maintain_radius_px 안의 최근접 검출을
      잇는다 — 먼 오검출로의 순간 점프 방지. 단 다른 검출이 신뢰도에서
      switch_conf_margin 이상 앞서고 "방금 나타났거나 움직이고 있으면" 그쪽으로
      갈아탄다 (오검출을 잡고 있다가 진짜 손이 들어온 경우의 자기 회복).
      움직임 조건 덕에 정지 오검출은 신뢰도가 높아도 추적을 훔치지 못한다.
    - 미검출이 miss_ttl_frames를 넘으면 추적을 버리고 획득부터 다시 한다
      (호출 측의 손 상태 무효화 정책과 같은 값을 쓰는 것을 권장).
    """

    def __init__(self, maintain_radius_px: float = 200.0, min_motion_px: float = 8.0,
                 min_motion_frames: int = 0, miss_ttl_frames: int = 30,
                 switch_conf_margin: float = 0.2) -> None:
        self.maintain_radius_px = maintain_radius_px
        self.min_motion_px = min_motion_px
        self.min_motion_frames = min_motion_frames
        self.miss_ttl_frames = miss_ttl_frames
        self.switch_conf_margin = switch_conf_margin
        self._last_center: tuple[float, float] | None = None
        self._missing = 0
        self._acq_tracks: list[dict] = []   # {"center": (x, y), "moving": 연속 이동 프레임 수}
        self._prev_centers: list[tuple[float, float]] = []   # 직전 프레임 검출 중심들

    def reset(self) -> None:
        self._last_center = None
        self._missing = 0
        self._acq_tracks = []
        self._prev_centers = []

    def _is_moving(self, center: tuple[float, float],
                   prev_centers: list[tuple[float, float]]) -> bool:
        """검출이 방금 나타났거나(직전 프레임에 대응 없음) 실제로 움직였는지 판정한다."""

        nearest = None
        for prev in prev_centers:
            distance = math.hypot(center[0] - prev[0], center[1] - prev[1])
            if nearest is None or distance < nearest:
                nearest = distance
        return nearest is None or nearest >= self.min_motion_px

    def select(self, centers: list[tuple[float, float]],
               confidences: list[float]) -> int | None:
        """이번 프레임 검출 중심들에서 손으로 채택할 인덱스를 고른다 (없으면 None)."""

        prev_centers = self._prev_centers
        self._prev_centers = list(centers)
        if not centers:
            self._missing += 1
            if self._missing > self.miss_ttl_frames:
                self._last_center = None
            self._acq_tracks = []   # 이동 연속성이 끊겼으므로 획득 이력도 버린다
            return None

        # 유지: 직전 채택 위치 근처의 최근접 검출을 잇는다.
        if self._last_center is not None:
            best = None
            best_distance = None
            for index, center in enumerate(centers):
                distance = math.hypot(center[0] - self._last_center[0],
                                      center[1] - self._last_center[1])
                if distance <= self.maintain_radius_px and (
                    best is None or distance < best_distance
                ):
                    best, best_distance = index, distance
            if best is not None:
                # 자기 회복: 다른 검출이 신뢰도에서 확실히 앞서고 "방금 나타났거나
                # 움직이고 있으면"(진짜 손이 들어온 경우) 그쪽으로 갈아탄다.
                # 움직임 조건이 없으면 정지 오검출이 신뢰도만으로 추적을 훔친다.
                strongest = max(range(len(centers)), key=lambda i: confidences[i])
                if (strongest != best
                        and confidences[strongest] >= confidences[best] + self.switch_conf_margin
                        and self._is_moving(centers[strongest], prev_centers)):
                    best = strongest
                self._last_center = centers[best]
                self._missing = 0
                return best
            self._missing += 1
            if self._missing > self.miss_ttl_frames:
                self._last_center = None   # 추적 폐기 → 아래에서 획득 시도
            else:
                return None

        # 획득: 기본은 최고 신뢰도 검출을 즉시 채택한다 (정지한 손도 잡힘).
        if self.min_motion_frames <= 0:
            chosen = max(range(len(centers)), key=lambda i: confidences[i])
            self._last_center = centers[chosen]
            self._missing = 0
            self._acq_tracks = []
            return chosen

        # (선택) 움직임 게이트: 연속으로 움직인 검출만 손으로 인정한다.
        chosen = None
        chosen_confidence = -1.0
        new_tracks: list[dict] = []
        for index, center in enumerate(centers):
            moving = 0
            best_track = None
            best_distance = None
            for track in self._acq_tracks:
                distance = math.hypot(center[0] - track["center"][0],
                                      center[1] - track["center"][1])
                if distance <= self.maintain_radius_px and (
                    best_track is None or distance < best_distance
                ):
                    best_track, best_distance = track, distance
            if best_track is not None and best_distance >= self.min_motion_px:
                moving = best_track["moving"] + 1
            new_tracks.append({"center": center, "moving": moving})
            if moving >= self.min_motion_frames and confidences[index] > chosen_confidence:
                chosen, chosen_confidence = index, confidences[index]
        self._acq_tracks = new_tracks
        if chosen is not None:
            self._last_center = centers[chosen]
            self._missing = 0
            self._acq_tracks = []
        return chosen


class HandTracker:
    """프레임에서 로봇손을 검출하고 HandState를 유지한다."""

    def __init__(self, config: HandTrackerConfig, repository_root: Path, model: Any | None = None) -> None:
        self.config = config
        self.repository_root = repository_root
        self.estimator = HandMotionEstimator(
            config.direction_ema_alpha, config.min_speed_px, config.hand_width_mm,
            width_px_decay=config.width_px_decay,
        )
        self.model = model if model is not None else self._load_model()
        self.frame_index = 0
        self.missing_frames = 0
        self.last_box: tuple[int, int, int, int] | None = None
        self.last_state: HandState | None = None

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repository_root / path

    def _load_model(self) -> Any:
        """백엔드에 맞는 검출기를 로드한다."""

        if self.config.backend == "motion":
            # 배경 차분: 고정 카메라·정적 물체 전제. 학습 불필요, 연산 비용 최소.
            import cv2

            self.background = cv2.createBackgroundSubtractorMOG2(
                history=self.config.motion_history,
                varThreshold=self.config.motion_var_threshold,
                detectShadows=False,
            )
            return None

        from ultralytics import YOLO

        if self.config.backend == "yolo_world":
            model = YOLO(str(self._resolve(self.config.weights)))
            model.set_classes(list(self.config.prompts))
            return model
        if self.config.backend == "custom":
            if not self.config.custom_weights:
                raise ValueError("backend=custom이면 custom_weights 경로가 필요합니다")
            return YOLO(str(self._resolve(self.config.custom_weights)))
        raise ValueError(f"지원하지 않는 손 추적 백엔드입니다: {self.config.backend}")

    def _detect_motion(self, frame) -> tuple[tuple[int, int, int, int], float] | None:
        """배경 차분으로 가장 큰 움직임 덩어리를 찾아 박스와 면적 비율을 반환한다."""

        import cv2
        import numpy as np

        blur = self.config.motion_blur
        prepared = cv2.GaussianBlur(frame, (blur, blur), 0) if blur > 1 else frame
        mask = self.background.apply(prepared, learningRate=self.config.motion_learning_rate)
        mask = (mask > 200).astype(np.uint8) * 255
        # 진단: 단계별 픽셀 수를 남겨 검출 실패 지점을 알 수 있게 한다 (scripts/hand_tracker_debug.py)
        debug = {"fg_px": int(np.count_nonzero(mask)), "gated_px": None,
                 "largest_area": 0.0, "min_area": self.config.motion_min_area_px,
                 "warmup_done": True, "reject": ""}
        self.last_debug = debug

        # 프레임 간 차분 게이팅: 배경 모델 잔상(ghost)은 지금 움직이지 않으므로 제거된다.
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        previous = getattr(self, "_previous_gray", None)
        self._previous_gray = gray
        motion_frames = getattr(self, "_motion_frames", 0) + 1
        self._motion_frames = motion_frames
        if motion_frames <= self.config.motion_warmup_frames:
            debug["warmup_done"] = False
            debug["reject"] = "warmup"
            return None
        if self.config.motion_use_framediff:
            if previous is None:
                debug["reject"] = "no_previous"
                return None
            diff = cv2.absdiff(gray, previous)
            moving = (diff > self.config.motion_framediff_threshold).astype(np.uint8) * 255
            d = self.config.motion_framediff_dilate
            moving = cv2.dilate(moving, np.ones((d, d), np.uint8))
            mask = cv2.bitwise_and(mask, moving)
            debug["gated_px"] = int(np.count_nonzero(mask))

        kernel = np.ones((self.config.motion_morph_kernel, self.config.motion_morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.last_motion_mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            debug["reject"] = "no_contour"
            return None
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        debug["largest_area"] = area
        if area < self.config.motion_min_area_px:
            debug["reject"] = "min_area"
            return None
        x, y, w, h = cv2.boundingRect(largest)
        self.last_contour = largest
        frame_area = float(frame.shape[0] * frame.shape[1])
        return (x, y, x + w, y + h), area / frame_area

    def _leading_point(self, contour, direction) -> tuple[float, float] | None:
        """이동 방향으로 가장 앞선 윤곽점(손끝 쪽)을 반환한다."""

        import numpy as np

        points = contour.reshape(-1, 2).astype(np.float32)
        projections = points[:, 0] * direction[0] + points[:, 1] * direction[1]
        best = points[int(np.argmax(projections))]
        return float(best[0]), float(best[1])

    def detect(self, frame) -> tuple[tuple[int, int, int, int], float] | None:
        """프레임에서 가장 신뢰도 높은 손 박스 (x1, y1, x2, y2)와 신뢰도를 반환한다."""

        if self.config.backend == "motion":
            return self._detect_motion(frame)

        result = self.model.predict(
            frame, imgsz=self.config.imgsz, conf=self.config.conf_threshold,
            device=self.config.device, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        best = None
        for (x1, y1, x2, y2), confidence in zip(boxes, confidences):
            if (x2 - x1) < self.config.min_box_px or (y2 - y1) < self.config.min_box_px:
                continue
            if best is None or confidence > best[1]:
                best = ((int(x1), int(y1), int(x2), int(y2)), float(confidence))
        return best

    def update(self, frame) -> HandState | None:
        """프레임마다 호출. detect_every 간격으로 검출하고 그 사이는 마지막 상태를 유지한다."""

        self.frame_index += 1
        if self.frame_index % self.config.detect_every == 0 or self.last_state is None:
            detection = self.detect(frame)
            if detection is None:
                self.missing_frames += self.config.detect_every
            else:
                (x1, y1, x2, y2), _ = detection
                self.last_box = (x1, y1, x2, y2)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                state = self.estimator.observe(center, float(x2 - x1))
                # motion 백엔드: 방향을 알면 덩어리의 선단(손끝 쪽)을 손 위치로 쓴다.
                # 사람 팔이 의수를 들고 들어와도 진행 방향 앞쪽이 의수다.
                if (
                    self.config.backend == "motion"
                    and self.config.motion_position_mode == "leading"
                    and state.direction is not None
                    and getattr(self, "last_contour", None) is not None
                ):
                    tip = self._leading_point(self.last_contour, state.direction)
                    if tip is not None:
                        state = HandState(position=tip, direction=state.direction,
                                          mm_per_px=state.mm_per_px)
                self.last_state = state
                self.missing_frames = 0
        if self.missing_frames >= self.config.max_missing_frames:
            # 오래된 위치로 결정하지 않도록 상태를 무효화한다.
            self.last_state = None
            self.last_box = None
            self.estimator.reset()
        return self.last_state
