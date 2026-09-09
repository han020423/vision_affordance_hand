"""실시간 affordance 분할 추론과 시각화.

학습된 YOLO11n-seg 가중치로 카메라 프레임 또는 단일 이미지에서
grasp_region(0)과 functional_region(1) 마스크를 추론한다. 물체 범주
이름은 사용하지 않고 affordance 클래스와 마스크만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import yaml

# 클래스 이름은 모델 checkpoint의 names를 그대로 따르고, 색과 파지 힌트만
# 이름 기준으로 정의한다. 2클래스(grasp/functional)와 파지 종류 3클래스
# (handle/body/functional) 모델을 모두 지원한다.
CLASS_COLORS_BY_NAME: dict[str, tuple[int, int, int]] = {
    "grasp_region": (80, 200, 80),          # 초록
    "handle_grasp_region": (80, 200, 80),   # 초록: 손잡이형(PRECISION/WRAP)
    "body_grasp_region": (230, 160, 60),    # 파랑: 몸통형(POWER)
    "functional_region": (60, 60, 230),     # 빨강: 회피
}
GRASP_HINT_BY_NAME: dict[str, str] = {
    "handle_grasp_region": "WRAP/PREC",
    "body_grasp_region": "POWER",
    "functional_region": "AVOID",
}
UNKNOWN_COLOR_BGR: tuple[int, int, int] = (0, 215, 255)  # 예상 밖 클래스: 노랑 경고색


@dataclass
class RealtimeConfig:
    """실시간 추론 설정. YAML 값을 CLI 인자가 덮어쓴 최종 값을 담는다."""

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    max_consecutive_failures: int = 30
    imgsz: int = 640
    conf_threshold: float = 0.5
    device: str = "cpu"
    window_name: str = "affordance realtime"
    overlay_alpha: float = 0.45
    output_dir: Path = Path("outputs/realtime_test")
    record_video: bool = False
    video_path: Path | None = None      # 설정 시 카메라 대신 영상 파일로 오프라인 검증
    display: bool = True                # False면 창을 띄우지 않는다 (녹화·배치 검증용)
    max_frames: int | None = None       # 지정 시 이 프레임 수만 처리하고 종료 (배치 검증용)


def load_realtime_config(path: Path) -> RealtimeConfig:
    """YAML 설정 파일을 읽어 RealtimeConfig로 해석한다."""

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    camera = dict(document.get("camera", {}))
    inference = dict(document.get("inference", {}))
    display = dict(document.get("display", {}))
    save = dict(document.get("save", {}))
    return RealtimeConfig(
        camera_index=int(camera.get("index", 0)),
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        max_consecutive_failures=int(camera.get("max_consecutive_failures", 30)),
        imgsz=int(inference.get("imgsz", 640)),
        conf_threshold=float(inference.get("conf_threshold", 0.5)),
        device=str(inference.get("device", "cpu")),
        window_name=str(display.get("window_name", "affordance realtime")),
        overlay_alpha=float(display.get("overlay_alpha", 0.45)),
        output_dir=Path(str(save.get("output_dir", "outputs/realtime_test"))),
        record_video=bool(save.get("record_video", False)),
    )


def render_affordance_overlay(
    frame_bgr: np.ndarray,
    result: Any,
    overlay_alpha: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """추론 결과의 마스크 폴리곤을 프레임 위에 채색하고 검출 요약을 반환한다.

    마스크 좌표는 Ultralytics가 원본 해상도로 되돌린 polygon(`masks.xy`)을
    사용하므로 letterbox 좌표 변환을 직접 계산하지 않는다.
    """

    overlay = frame_bgr.copy()
    detections: list[dict[str, Any]] = []
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return overlay, detections

    color_layer = overlay.copy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    model_names = {int(key): str(value) for key, value in getattr(result, "names", {}).items()}
    for polygon, class_id, confidence in zip(result.masks.xy, class_ids, confidences):
        # 아주 작은 검출은 polygon 점이 부족할 수 있으므로 건너뛴다.
        if polygon is None or len(polygon) < 3:
            continue
        class_name = model_names.get(int(class_id), f"class_{class_id}")
        color = CLASS_COLORS_BY_NAME.get(class_name, UNKNOWN_COLOR_BGR)
        points = polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(color_layer, [points], color)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
        # 신뢰도 문구는 폴리곤 가장 위쪽 점 근처에 표시한다.
        top_point = points[points[:, 0, 1].argmin(), 0]
        hint = GRASP_HINT_BY_NAME.get(class_name)
        label = f"{class_name} {confidence:.2f}" + (f" [{hint}]" if hint else "")
        cv2.putText(
            overlay,
            label,
            (int(top_point[0]), max(18, int(top_point[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "grasp_hint": hint,
                "confidence": float(confidence),
                "polygon_points": int(len(polygon)),
            }
        )
    cv2.addWeighted(color_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)
    return overlay, detections


def draw_selection(overlay: np.ndarray, decision, hand) -> None:
    """후보 선택 결과(파지점·순위·상태·모의 손)를 화면에 그린다."""

    for rank, candidate in enumerate(decision.ranked):
        x, y = candidate.grasp_point
        is_top = decision.state == "GRASP" and candidate is decision.candidate
        color = (255, 255, 255) if is_top else (180, 180, 180)
        cv2.circle(overlay, (x, y), 10 if is_top else 6, color, 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 2, color, -1, cv2.LINE_AA)
        label = f"#{rank + 1} {candidate.scores['total']:.2f}"
        if is_top:
            label += f" -> {decision.pose}"
        if candidate.scores.get("point_held"):
            label += " HOLD"   # 가림 중이라 마지막 검증 파지점을 유지하고 있음
        cv2.putText(overlay, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2 if is_top else 1, cv2.LINE_AA)

    # 모의 손 표시: 십자 + 접근 방향 화살표
    hx, hy = int(hand.position[0]), int(hand.position[1])
    cv2.drawMarker(overlay, (hx, hy), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
    if hand.direction is not None:
        tip = (int(hx + hand.direction[0] * 55), int(hy + hand.direction[1] * 55))
        cv2.arrowedLine(overlay, (hx, hy), tip, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    # cv2.putText는 한글을 렌더링하지 못해 ?????? 로 깨진다 → 영문 표기 사용.
    banner = {"GRASP": f"GRASP: {decision.pose}", "ALIGN": "ALIGN (adjust hand)",
              "NO_TARGET": "NO TARGET"}[decision.state]
    banner_color = {"GRASP": (80, 200, 80), "ALIGN": (0, 200, 255),
                    "NO_TARGET": (120, 120, 120)}[decision.state]
    cv2.putText(overlay, banner, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                banner_color, 2, cv2.LINE_AA)


def draw_hand_status(overlay: np.ndarray, status) -> None:
    """손 제어 상태 머신의 현재 상태를 화면 아래쪽에 표시한다."""

    colors = {
        "SEARCH": (180, 180, 180),
        "ALIGN": (0, 200, 255),
        "TARGET_SELECTED": (80, 200, 80),
        "PRE_SHAPE": (80, 200, 200),
        "CLOSE": (60, 160, 255),
        "HOLD": (80, 220, 120),
        "OPEN": (200, 200, 80),
        "FAULT": (60, 60, 230),
    }
    color = colors.get(status.state, (255, 255, 255))
    height = overlay.shape[0]
    cv2.putText(overlay, f"HAND {status.text}", (12, height - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, "space=grasp  o=open  x=stop", (12, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def infer_single_image(
    model: Any,
    image_path: Path,
    config: RealtimeConfig,
    output_path: Path,
) -> list[dict[str, Any]]:
    """카메라 없이 단일 이미지로 모델과 시각화를 검증한다."""

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    result = model.predict(
        frame,
        imgsz=config.imgsz,
        conf=config.conf_threshold,
        device=config.device,
        verbose=False,
    )[0]
    overlay, detections = render_affordance_overlay(frame, result, config.overlay_alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"overlay 저장에 실패했습니다: {output_path}")
    return detections


def _open_camera(config: RealtimeConfig) -> cv2.VideoCapture:
    """플랫폼에 맞는 백엔드로 카메라(또는 오프라인 검증용 영상 파일)를 연다."""

    import platform

    video_path = getattr(config, "video_path", None)
    if video_path:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")
        return capture
    # Windows에서는 DirectShow가 기본 MSMF보다 초기화가 안정적인 경우가 많다.
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    capture = cv2.VideoCapture(config.camera_index, backend)
    if not capture.isOpened():
        raise RuntimeError(
            f"카메라 {config.camera_index}번을 열 수 없습니다. "
            "장치 번호와 연결 상태를 확인하세요."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    return capture


def run_camera_loop(
    model: Any,
    config: RealtimeConfig,
    selection_config: dict[str, Any] | None = None,
    hand_tracker: Any | None = None,
    hand_controller: Any | None = None,
) -> dict[str, Any]:
    """카메라 프레임을 반복 추론하고 overlay 창을 표시한다.

    조작 키:
      q 또는 ESC: 종료
      s: 현재 원본과 overlay 스냅숏 저장
      hand_controller가 있을 때:
        스페이스: 파지(CLOSE) 트리거   o: 펼침(OPEN)   x: 비상정지(STOP)

    selection_config가 주어지면 후보 선택을 함께 수행한다. ArUco 실측 전이므로
    마우스 커서를 모의 손 위치로, 커서 이동 방향을 접근 방향으로 사용한다.

    연속 프레임 읽기 실패가 max_consecutive_failures를 넘으면 카메라 끊김으로
    판단하고 안전하게 종료한다. 반환값은 실행 요약이다.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    capture = _open_camera(config)

    hand_state = None
    suppression_filter = None
    if selection_config is not None:
        from src.grasp_selection import (
            HandState,
            HandSuppressionFilter,
            decide_grasp,
            extract_candidates,
        )

        # 로봇손 오검출 억제 (시간 안정성 + 추적 박스 겹침 거부)
        suppression_filter = HandSuppressionFilter.from_config(selection_config)

    # 손 위치 소스: 손 추적기(로봇손 검출)가 있으면 그것을, 없으면 마우스 모의 손을 쓴다.
    if selection_config is not None and hand_tracker is None:
        hand_state = HandState(position=(config.width / 2, config.height / 2))
        direction_ema = [0.0, 0.0]

        def on_mouse(event, x, y, flags, param):  # noqa: ANN001
            nonlocal hand_state
            dx = x - hand_state.position[0]
            dy = y - hand_state.position[1]
            # 커서 이동 방향의 지수이동평균을 접근 방향으로 사용한다.
            if abs(dx) + abs(dy) > 1:
                direction_ema[0] = 0.8 * direction_ema[0] + 0.2 * dx
                direction_ema[1] = 0.8 * direction_ema[1] + 0.2 * dy
            norm = (direction_ema[0] ** 2 + direction_ema[1] ** 2) ** 0.5
            direction = (
                (direction_ema[0] / norm, direction_ema[1] / norm) if norm > 3 else None
            )
            hand_state = HandState(position=(float(x), float(y)), direction=direction)

        cv2.namedWindow(config.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(config.window_name, on_mouse)
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    snapshot_count = 0
    failure_streak = 0
    fps_smoothed = 0.0
    stop_reason = "user_quit"
    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                if config.video_path:
                    stop_reason = "video_finished"
                    break
                # 일시적 끊김은 허용하되 연속 실패가 길어지면 안전 종료한다.
                failure_streak += 1
                if failure_streak >= config.max_consecutive_failures:
                    stop_reason = "camera_disconnected"
                    break
                continue
            failure_streak = 0

            result = model.predict(
                frame,
                imgsz=config.imgsz,
                conf=config.conf_threshold,
                device=config.device,
                verbose=False,
            )[0]
            overlay, _ = render_affordance_overlay(frame, result, config.overlay_alpha)

            if hand_tracker is not None:
                hand_state = hand_tracker.update(frame)
                if hand_tracker.last_box is not None:
                    x1, y1, x2, y2 = hand_tracker.last_box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(overlay, "robot hand", (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                elif selection_config is not None:
                    cv2.putText(overlay, "HAND NOT DETECTED", (12, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)

            decision = None
            if selection_config is not None and hand_state is not None:
                candidates, functional_mask = extract_candidates(
                    result,
                    frame.shape,
                    min_confidence=selection_config["candidate"]["min_confidence"],
                    min_area_px=selection_config["candidate"]["min_area_px"],
                    boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
                )
                if suppression_filter is not None:
                    # 손 박스는 추적이 살아 있을 때만 쓴다(스테일 박스로 진짜 목표를 죽이지 않기 위해).
                    hand_box = (
                        hand_tracker.last_box
                        if hand_tracker is not None and hand_tracker.last_box is not None
                        else None
                    )
                    candidates, _ = suppression_filter.update(candidates, hand_box=hand_box)
                decision = decide_grasp(candidates, functional_mask, hand_state, selection_config)
                draw_selection(overlay, decision, hand_state)

            # 손 제어: 추적이 끊긴 프레임도 알려야 파지 직전 중단이 동작한다.
            if hand_controller is not None:
                hand_status = hand_controller.update(
                    decision, hand_valid=hand_state is not None
                )
                draw_hand_status(overlay, hand_status)

            elapsed = time.perf_counter() - started
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            # 지수이동평균으로 FPS 표기의 흔들림을 줄인다.
            fps_smoothed = instant_fps if frame_count == 0 else 0.9 * fps_smoothed + 0.1 * instant_fps
            cv2.putText(
                overlay,
                f"FPS {fps_smoothed:5.1f}  conf>={config.conf_threshold:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if config.record_video:
                if writer is None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = config.output_dir / f"realtime_overlay_{stamp}.mp4"
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        15.0,
                        (overlay.shape[1], overlay.shape[0]),
                    )
                writer.write(overlay)

            frame_count += 1
            if config.max_frames is not None and frame_count >= config.max_frames:
                stop_reason = "max_frames"
                break
            if not config.display:
                continue
            cv2.imshow(config.window_name, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if hand_controller is not None:
                if key == ord(" "):
                    hand_controller.request_close()
                elif key == ord("o"):
                    hand_controller.request_open()
                elif key == ord("x"):
                    hand_controller.emergency_stop()
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(config.output_dir / f"snapshot_{stamp}_raw.png"), frame)
                cv2.imwrite(str(config.output_dir / f"snapshot_{stamp}_overlay.png"), overlay)
                snapshot_count += 1
    finally:
        # 어떤 경로로 끝나도 손을 먼저 세운다.
        if hand_controller is not None:
            try:
                hand_controller.emergency_stop()
            except Exception:
                pass
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    summary = {
        "frames_processed": frame_count,
        "snapshots_saved": snapshot_count,
        "stop_reason": stop_reason,
        "last_fps": round(fps_smoothed, 1),
    }
    if hand_controller is not None:
        summary["hand_state"] = hand_controller.status.state
        summary["hand_detail"] = hand_controller.status.detail
    return summary
