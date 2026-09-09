#!/usr/bin/env python
"""RF-DETR-Seg 체크포인트로 실시간 affordance 분할과 파지 후보 선택을 시험한다.

run_realtime_seg.py(Ultralytics 전용)와 같은 카메라·선택 설정 YAML을 공유한다.
RF-DETR 3클래스(handle/body/functional) 추론과 FPS 계측에 더해, --select를 주면
기존 파지 후보 선택 파이프라인(src/grasp_selection)에 연결한다. 카메라 모드에서는
마우스 커서가 모의 손이고, --image 단일 이미지 모드는 결정 JSON을 출력한다
(run_realtime_seg.py의 같은 모드와 출력 형식이 같다).

예시:
    python scripts/run_realtime_rfdetr.py \
      --model outputs/training/custom_finetune_m_rfdetr_customv3_seed42/checkpoint_best_ema.pth \
      --select

조작 키: q/ESC 종료, s 스냅숏 저장
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.perception.realtime import load_realtime_config  # noqa: E402

# 클래스 ID(0시작) -> (이름, BGR 색). 검수 오버레이와 같은 색 체계를 쓴다.
CLASSES = {
    0: ("handle", (0, 200, 0)),
    1: ("body", (255, 120, 0)),
    2: ("functional", (0, 0, 255)),
    3: ("robot_hand", (0, 255, 255)),   # 실험 N(v4)부터. 파지 후보가 아니라 손 추적 소스
}
ROBOT_HAND_CLASS = 3


def _weights_device(model) -> str:
    """rfdetr 래퍼 안의 실제 파라미터가 올라간 장치를 문자열로 돌려준다."""

    for attribute in ("model.model", "model"):
        target = model
        for part in attribute.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None or not hasattr(target, "parameters"):
            continue
        for parameter in target.parameters():
            return str(parameter.device)
    return "unknown"


def resolve_repo_path(value: str) -> Path:
    """저장소 상대경로를 절대경로로 해석한다."""

    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def render_overlay(
    frame: np.ndarray, detections, class_id_base: int, alpha: float,
    robot_hand_min_conf: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """RF-DETR 예측 마스크를 프레임에 채색하고 클래스별 검출 수를 센다.

    robot_hand_min_conf: robot_hand 마스크의 표시 하한. 물체 위에 간헐적으로 뜨는
    저신뢰(0.2대) robot_hand 중복 검출이 화면을 깜빡이게 하는 문제(2026-08-26,
    컵에서 11/98프레임 확인)의 표시 안정화다. 판정 파이프라인은 이 검출을 신뢰도
    비교 억제·추적 하한(0.5)으로 이미 무해화하고 있으므로 표시만 걸러도 안전하다.
    """

    counts = {name: 0 for name, _ in CLASSES.values()}
    if detections is None or detections.mask is None or len(detections) == 0:
        return frame, counts
    output = frame.astype(np.float32)
    height, width = frame.shape[:2]
    for mask, raw_id, confidence in zip(
        np.asarray(detections.mask),
        np.asarray(detections.class_id, dtype=int),
        np.asarray(detections.confidence),
    ):
        class_id = int(raw_id) - class_id_base
        if class_id not in CLASSES:
            continue
        if class_id == ROBOT_HAND_CLASS and float(confidence) < robot_hand_min_conf:
            continue
        name, color = CLASSES[class_id]
        counts[name] += 1
        mask = mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        region = mask.astype(bool)
        output[region] = output[region] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
        if contours:
            x, y, _, _ = cv2.boundingRect(contours[0])
            cv2.putText(
                output, f"{name} {confidence:.2f}", (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
    return np.clip(output, 0, 255).astype(np.uint8), counts


def detections_to_arrays(
    detections, class_id_base: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rfdetr 예측(supervision Detections 호환)을 배열 3개로 변환한다.

    반환: (masks (N,H,W) bool, class_ids (N,) int 0기준, confidences (N,) float).
    class_id_base를 빼서 0=handle/1=body/2=functional 규약으로 정규화한다.
    mask/class_id/confidence 속성만 사용하므로(덕 타이핑) rfdetr 없이 시험 가능하다.
    검출이 없거나 mask가 없으면 세 배열 모두 길이 0으로 돌려준다.
    """

    empty = (
        np.zeros((0, 0, 0), dtype=bool),
        np.zeros((0,), dtype=int),
        np.zeros((0,), dtype=float),
    )
    if detections is None or getattr(detections, "mask", None) is None:
        return empty
    masks = np.asarray(detections.mask)
    if masks.size == 0 or masks.ndim != 3:
        return empty
    class_ids = np.asarray(detections.class_id, dtype=int) - int(class_id_base)
    confidences = np.asarray(detections.confidence, dtype=float)
    if not (len(masks) == len(class_ids) == len(confidences)):
        raise ValueError(
            "검출 배열 길이가 어긋납니다: "
            f"mask {len(masks)}, class_id {len(class_ids)}, confidence {len(confidences)}"
        )
    return masks.astype(bool), class_ids, confidences


def run_single_image(args, config, model, selection_config, output_dir: Path) -> int:
    """카메라 없이 단일 이미지로 추론(선택 포함)을 검증하고 결과 JSON을 출력한다.

    --select가 있으면 run_realtime_seg.py의 단일 이미지 선택 모드와 같은
    JSON 키(state/pose/reason/candidates)를 출력해 두 모델의 결정을 비교할 수 있다.
    """

    image_path = resolve_repo_path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    detections = model.predict(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=config.conf_threshold
    )
    # class_id 기준(0시작이 실측 기본). min만으로 판정하면 body(1)만 검출된
    # 프레임에서 1시작으로 오판하므로, 0시작이면 나올 수 없는 id(>3) 관측 시에만
    # 1시작으로 본다 (2026-08-26 버그 수정 — 아래 실시간 루프 주석 참조).
    class_id_base = 0
    if detections is not None and len(detections) > 0:
        observed = np.asarray(detections.class_id, dtype=int)
        if int(observed.max()) > ROBOT_HAND_CLASS and int(observed.min()) != 0:
            class_id_base = 1
    overlay, counts = render_overlay(frame, detections, class_id_base, config.overlay_alpha,
                                     robot_hand_min_conf=args.hand_seg_conf)

    if args.image_output is not None:
        output_path = resolve_repo_path(args.image_output)
    else:
        output_path = output_dir / f"rfdetr_single_overlay_{image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if selection_config is None:
        cv2.imwrite(str(output_path), overlay)
        print(json.dumps(
            {"mode": "single_image", "model": str(resolve_repo_path(args.model)),
             "image": str(image_path), "overlay_saved": str(output_path),
             "class_id_base": class_id_base, "detections": counts},
            ensure_ascii=False, indent=2))
        return 0

    from src.grasp_selection import HandState, decide_grasp, extract_candidates_from_arrays
    from src.perception.realtime import draw_selection

    hand_position = (
        args.hand_x if args.hand_x is not None else frame.shape[1] * 0.15,
        args.hand_y if args.hand_y is not None else frame.shape[0] * 0.5,
    )
    # 정지 이미지 모드에서는 화면 중심을 향하는 방향을 접근 방향으로 가정한다.
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    vector = (center[0] - hand_position[0], center[1] - hand_position[1])
    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    direction = (vector[0] / norm, vector[1] / norm) if norm > 1e-6 else None
    hand = HandState(position=hand_position, direction=direction)

    masks, class_ids, confidences = detections_to_arrays(detections, class_id_base)
    candidates, functional_mask = extract_candidates_from_arrays(
        masks, class_ids, confidences, frame.shape,
        min_confidence=selection_config["candidate"]["min_confidence"],
        min_area_px=selection_config["candidate"]["min_area_px"],
        boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
    )
    decision = decide_grasp(candidates, functional_mask, hand, selection_config)
    draw_selection(overlay, decision, hand)
    cv2.imwrite(str(output_path), overlay)
    print(json.dumps(
        {"mode": "single_image_select", "state": decision.state,
         "pose": decision.pose, "reason": decision.reason,
         "overlay_saved": str(output_path),
         "candidates": [
             {"class": c.class_name, "point": c.grasp_point,
              "width_px": round(c.width_px, 1),
              "total": round(float(c.scores["total"]), 3)}
             for c in decision.ranked
         ]},
        ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="outputs/training/custom_finetune_k_rfdetr_seg_seed42/checkpoint_best_ema.pth",
        help="RF-DETR-Seg 체크포인트 경로",
    )
    parser.add_argument("--variant", default="RFDETRSegNano", help="rfdetr 모델 변형 이름")
    parser.add_argument("--config", default="configs/hardware/realtime_camera.yaml")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="자동 종료까지 초(0=q 키로만 종료)")
    parser.add_argument("--record", action="store_true", help="overlay 영상을 녹화한다")
    parser.add_argument("--no-window", action="store_true", help="창 없이 계측만 수행한다")
    parser.add_argument(
        "--optimize", action="store_true",
        help="추론 최적화(optimize_for_inference)를 시도한다. 실패하면 원본 그대로 진행",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="추론 장치. auto=CUDA 가능하면 cuda:0, 아니면 cpu. 예: cuda:0, cpu",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="후보 선택을 함께 수행한다 (카메라 모드에서는 마우스가 모의 손)",
    )
    parser.add_argument(
        "--selection-config",
        default="configs/grasp_selection.yaml",
        help="후보 선택 설정 YAML",
    )
    parser.add_argument(
        "--human-prior-config",
        default="configs/human_grasp_prior.yaml",
        help="인간 파지 사전확률 YAML",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="카메라 대신 이 이미지 한 장으로 추론(선택 포함)을 검증하고 overlay를 저장한다",
    )
    parser.add_argument(
        "--image-output",
        default=None,
        help="--image 모드의 overlay 저장 경로(기본: 출력 폴더 아래 자동 이름)",
    )
    parser.add_argument("--hand-x", type=float, default=None, help="--image 모드 모의 손 x")
    parser.add_argument("--hand-y", type=float, default=None, help="--image 모드 모의 손 y")
    parser.add_argument(
        "--hand-source",
        choices=["mouse", "detector", "segmentation"],
        default="mouse",
        help="손 위치 소스: mouse(모의 손) / detector(배경차분 추적) / "
             "segmentation(분할 모델의 robot_hand 클래스 — 실험 N 이상 필요)",
    )
    parser.add_argument(
        "--hand-tracker-config",
        default="configs/hardware/hand_tracker.yaml",
        help="로봇손 추적 설정 YAML (--hand-source detector에서 사용)",
    )
    parser.add_argument(
        "--hand-seg-conf",
        type=float,
        default=0.5,
        help="robot_hand 검출을 손 추적(중심·스케일)에 쓰는 최소 신뢰도. "
             "이보다 낮은 검출은 픽셀 억제(신뢰도 비교)에만 참여한다",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Brunel Hand 컨트롤러 시리얼 포트(예: /dev/ttyACM0). 주면 실제 손을 구동한다",
    )
    parser.add_argument(
        "--hand-mock",
        action="store_true",
        help="실제 손 없이 모의 컨트롤러로 전체 흐름을 점검한다",
    )
    parser.add_argument(
        "--hand-config",
        default="configs/hardware/hand_actuators.yaml",
        help="손 보정값·프리셋 YAML",
    )
    parser.add_argument(
        "--hand-auto-close",
        action="store_true",
        help="후보가 안정되면 사용자 트리거 없이 파지한다(기본은 스페이스 키 필요)",
    )
    parser.add_argument(
        "--hand-stable-frames",
        type=int,
        default=5,
        help="같은 후보·자세가 이 프레임 수 이상 유지되면 파지 준비로 넘어간다",
    )
    args = parser.parse_args()

    config = load_realtime_config(resolve_repo_path(args.config))
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.conf is not None:
        config.conf_threshold = args.conf
    output_dir = resolve_repo_path(str(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_repo_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {model_path}")

    import torch
    import rfdetr

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA를 쓸 수 없습니다. torch가 GPU를 못 잡았거나 드라이버 문제입니다. "
            "CPU로 강제하려면 --device cpu를 지정하세요."
        )
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(compute capability {'.'.join(map(str, torch.cuda.get_device_capability(0)))})")

    if not hasattr(rfdetr, args.variant):
        raise ValueError(f"rfdetr에 없는 모델 변형입니다: {args.variant}")
    print(f"모델 로딩 중: {args.variant} ({model_path.name}), 요청 장치={device}")
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(model_path), device=device)
    if args.optimize:
        try:
            model.optimize_for_inference()
            print("추론 최적화 적용")
        except Exception as error:  # noqa: BLE001 - 최적화 실패는 치명적이지 않다
            print(f"추론 최적화 생략(실패): {error!r}")

    selection_config = None
    if args.select:
        from src.grasp_selection import load_selection_config

        selection_config = load_selection_config(
            resolve_repo_path(args.selection_config),
            resolve_repo_path(args.human_prior_config),
        )

    if args.image is not None:
        return run_single_image(args, config, model, selection_config, output_dir)

    hand_client = None
    hand_controller = None
    if args.hand_port or args.hand_mock:
        # 손 제어는 파지 결정(Decision)이 있어야 하므로 --select가 전제다.
        if selection_config is None:
            print("손 제어는 --select와 함께 써야 합니다(파지 결정이 필요).", file=sys.stderr)
            return 2
        from datetime import datetime

        from src.hand_control import (
            GraspController,
            HandClient,
            MockLink,
            SerialLink,
            load_calibration,
        )
        from src.perception.realtime import draw_hand_status

        hand_calibration = load_calibration(resolve_repo_path(args.hand_config))
        hand_log_dir = REPOSITORY_ROOT / "outputs" / "hand_tests"
        hand_log_dir.mkdir(parents=True, exist_ok=True)
        hand_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hand_log_path = hand_log_dir / f"realtime_hand_{hand_stamp}.log"
        hand_log = hand_log_path.open("a", encoding="utf-8")

        def record(text: str) -> None:
            stamp_now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{stamp_now} {text}", file=hand_log, flush=True)

        if args.hand_mock:
            link = MockLink(hand_calibration)
            print("손 제어: 모의 컨트롤러(MockLink)")
        else:
            link = SerialLink(args.hand_port, hand_calibration.baud)
            print(f"손 제어: {link.description}")
        hand_client = HandClient(link, hand_calibration, event_log=record)
        identity = hand_client.connect()
        print(f"손 펌웨어 확인: {identity.get('name')} fw {identity.get('fw')}")
        hand_controller = GraspController(
            hand_client,
            hand_calibration,
            stable_frames=args.hand_stable_frames,
            auto_close=args.hand_auto_close,
            log=record,
        )
        print(f"손 제어 로그: {hand_log_path}")
        print("조작: 스페이스=파지, o=펼침, x=비상정지")

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(config.camera_index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if not capture.isOpened():
        raise RuntimeError(f"카메라 {config.camera_index}번을 열 수 없습니다.")

    window_name = "rfdetr realtime"
    hand_state = None
    hand_tracker = None
    suppression_filter = None
    direction_ema = [0.0, 0.0]
    if selection_config is not None:
        from src.grasp_selection import (
            HandState,
            HandSuppressionFilter,
            decide_grasp,
            extract_candidates_from_arrays,
        )
        from src.perception.realtime import draw_selection

        # 로봇손 오검출 억제: 몇 프레임 연속 같은 자리의 후보만 물체로 인정한다.
        # 마우스 모의 손 모드에서는 손 박스가 없으므로 시간 안정성만 동작한다.
        suppression_filter = HandSuppressionFilter.from_config(selection_config)

        if args.hand_source == "segmentation":
            # 분할 모델의 robot_hand 마스크에서 손 위치를 얻는다. 방향은 위치 변화의
            # 지수이동평균(배경차분 추적기와 같은 방식·같은 설정값).
            from src.perception.hand_tracker import (
                HandMotionEstimator,
                SegmentationHandSelector,
                load_hand_tracker_config,
            )

            tracker_config = load_hand_tracker_config(resolve_repo_path(args.hand_tracker_config))
            hand_estimator = HandMotionEstimator(
                ema_alpha=tracker_config.direction_ema_alpha,
                min_speed_px=tracker_config.min_speed_px,
                hand_width_mm=tracker_config.hand_width_mm,
                width_px_decay=tracker_config.width_px_decay,
            )
            # 손 채택 규칙: 최고 신뢰도 검출을 즉시 획득(정지한 손 포함), 추적 중에는
            # 직전 위치 근처만 이어 먼 오검출로의 점프를 막고, 확실히 더 강한 검출이
            # 나타나면 갈아탄다. (움직임 게이트는 정지한 손을 못 잡아 기본 비활성 —
            # 2026-08-25 사용자 결정.)
            hand_selector = SegmentationHandSelector(
                miss_ttl_frames=tracker_config.max_missing_frames
            )
            hand_missing_streak = 0
            if not args.no_window:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            print("후보 선택 모드: 분할 모델 robot_hand 클래스로 손 추적")
        elif args.hand_source == "detector":
            # 로봇손 비전 추적: 손 위치·접근 방향을 실제 검출로 얻는다.
            # 추적 박스는 억제 필터의 손 겹침 거부에도 쓰인다.
            from src.perception.hand_tracker import HandTracker, load_hand_tracker_config

            tracker_config = load_hand_tracker_config(resolve_repo_path(args.hand_tracker_config))
            hand_tracker = HandTracker(tracker_config, REPOSITORY_ROOT)
            if not args.no_window:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            print(
                f"후보 선택 모드: 로봇손 비전 추적 "
                f"({tracker_config.backend}, 매 {tracker_config.detect_every}프레임)"
            )
        elif args.no_window:
            # ArUco 실측 전이므로 마우스 커서를 모의 손으로 쓴다(run_camera_loop와 동일).
            hand_state = HandState(position=(config.width / 2, config.height / 2))
            print("후보 선택 모드(--no-window): 모의 손은 화면 중앙에 고정된다")
        else:
            hand_state = HandState(position=(config.width / 2, config.height / 2))
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

            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, on_mouse)
            print("후보 선택 모드: 마우스 커서 = 모의 손 위치, 커서 이동 방향 = 접근 방향")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    writer = None
    inference_times: list[float] = []
    total_counts = {name: 0 for name, _ in CLASSES.values()}
    frames = 0
    failure_streak = 0
    class_id_base = 0  # 평가에서 확인된 값(0시작). 확정 증거가 나오면 고정한다.
    class_id_base_locked = False
    device_actual = "unknown"
    stop_reason = "duration_elapsed" if args.duration > 0 else "user_quit"
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                failure_streak += 1
                if failure_streak >= config.max_consecutive_failures:
                    stop_reason = "camera_disconnected"
                    break
                continue
            failure_streak = 0

            t0 = time.perf_counter()
            detections = model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=config.conf_threshold
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            inference_times.append(elapsed_ms)

            if frames == 0:
                device_actual = _weights_device(model)
                print(f"실제 가중치 장치: {device_actual}")
                if device.startswith("cuda") and not device_actual.startswith("cuda"):
                    raise RuntimeError(
                        f"GPU를 요청했지만 가중치가 {device_actual}에 있습니다. 중단합니다."
                    )

            if detections is not None and len(detections) > 0 and not class_id_base_locked:
                # 기준(base)은 확정 증거가 나올 때까지 0(실측 기본)을 쓰고, 나오면
                # 고정한다. 이전처럼 매 프레임 min으로 추정하면 body(1)만 검출된
                # 프레임에서 1시작으로 오판해 모든 클래스가 한 칸 밀린다:
                # body→handle 표시·판정, robot_hand→functional(손 미검출 처리).
                # 2026-08-26 실물에서 원통이 handle로 뜨던 주 원인이자, 그간
                # "손=functional, 클래스 요동" 증상의 숨은 원인이었다. handle(0)이
                # 잡힌 프레임만 정상이라 머그 장면에서는 늘 정상으로 보였다.
                observed = np.asarray(detections.class_id, dtype=int)
                if int(observed.min()) == 0:
                    class_id_base = 0            # 0 관측 = 0시작 확정
                    class_id_base_locked = True
                elif int(observed.max()) > ROBOT_HAND_CLASS:
                    class_id_base = 1            # 0시작이면 불가능한 id(4) 관측 = 1시작 확정
                    class_id_base_locked = True

            overlay, counts = render_overlay(frame, detections, class_id_base, config.overlay_alpha,
                                             robot_hand_min_conf=args.hand_seg_conf)
            for name, value in counts.items():
                total_counts[name] += value
            frames += 1

            if hand_tracker is not None:
                hand_state = hand_tracker.update(frame)
                if hand_tracker.last_box is not None:
                    bx1, by1, bx2, by2 = hand_tracker.last_box
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                    cv2.putText(overlay, "robot hand", (bx1, max(18, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                elif selection_config is not None:
                    cv2.putText(overlay, "HAND NOT DETECTED", (10, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            hand_box = None
            hand_conf_map = None
            if selection_config is not None and args.hand_source == "segmentation":
                masks_all, ids_all, confs_all = detections_to_arrays(detections, class_id_base)
                hand_select = ids_all == ROBOT_HAND_CLASS
                if hand_select.any():
                    # 픽셀별 robot_hand 최대 신뢰도 지도: 아래 신뢰도 비교 억제에 쓴다.
                    hand_conf_map = np.zeros(masks_all.shape[1:], dtype=np.float32)
                    for hand_mask, hand_conf in zip(masks_all[hand_select], confs_all[hand_select]):
                        np.maximum(hand_conf_map, hand_mask * float(hand_conf), out=hand_conf_map)
                # 손 추적(중심·스케일)은 고신뢰 검출만 쓴다: 물체 손잡이에 뜬 저신뢰
                # robot_hand 오검출(O 실물 시험의 0.3대)이 작은 박스를 만들면 mm/px
                # 환산이 오염되어 폭 필터가 정상 후보를 죽이는 연쇄가 생긴다.
                # 고신뢰 검출 중 SegmentationHandSelector가 고른 하나(와 그 주변 파편)만
                # 손으로 채택한다: 추적 중에는 직전 위치 근처만 이어 검은 정지 물체
                # 오검출로의 점프를 막고, 확실히 더 강한 검출이 나타나면 갈아탄다.
                track_ids = np.flatnonzero(hand_select & (confs_all >= args.hand_seg_conf))
                hand_infos = []   # (검출 인덱스, 중심, 박스)
                for i in track_ids:
                    ys, xs = np.nonzero(masks_all[i])
                    if len(xs) == 0:
                        continue
                    hand_infos.append((int(i), (float(xs.mean()), float(ys.mean())),
                                       (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
                # 검출이 없는 프레임도 select를 호출해야 선택기의 TTL이 진행된다.
                pick = hand_selector.select(
                    [info[1] for info in hand_infos],
                    [float(confs_all[info[0]]) for info in hand_infos],
                )
                chosen = hand_infos[pick] if pick is not None else None
                if chosen is not None:
                    hand_missing_streak = 0
                    # 손이 파편으로 나뉘어 검출되면 채택 검출 주변의 파편을 합쳐 박스를
                    # 만든다 (파편 하나만 쓰면 박스가 작아져 mm/px가 부풀기 때문).
                    x1, y1, x2, y2 = chosen[2]
                    merge_margin = 0.6 * max(x2 - x1 + 1, y2 - y1 + 1)
                    hand_track_union = masks_all[chosen[0]].copy()
                    for info in hand_infos:
                        if info is chosen:
                            continue
                        cx, cy = info[1]
                        if (x1 - merge_margin <= cx <= x2 + merge_margin
                                and y1 - merge_margin <= cy <= y2 + merge_margin):
                            hand_track_union |= masks_all[info[0]]
                    ys, xs = np.nonzero(hand_track_union)
                    hand_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                    center = (float(xs.mean()), float(ys.mean()))
                    # mm 환산용 폭은 박스의 짧은 변: 실측 손 가로(120mm)에 대응 (회전에 덜 민감)
                    box_short = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
                    hand_state = hand_estimator.observe(center, box_short)
                    cv2.rectangle(overlay, hand_box[:2], hand_box[2:], (0, 255, 255), 2)
                    cv2.putText(overlay, "robot hand", (hand_box[0], max(18, hand_box[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    # 잠깐의 검출 실패는 용서하되 오래 끊기면 무효화한다 (배경차분 추적기와 동일 정책).
                    hand_missing_streak += 1
                    if hand_missing_streak > tracker_config.max_missing_frames:
                        hand_state = None
                        hand_estimator.reset()
                    if hand_state is None:
                        cv2.putText(overlay, "HAND NOT DETECTED", (10, 62),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            decision = None
            if selection_config is not None and hand_state is not None:
                masks, class_ids, confidences = detections_to_arrays(detections, class_id_base)
                # robot_hand는 파지 후보가 아니다: 후보 추출 전에 분리한다.
                keep = class_ids != ROBOT_HAND_CLASS
                masks, class_ids, confidences = masks[keep], class_ids[keep], confidences[keep]
                object_detections = len(masks)          # 억제·conf/면적 컷 이전의 물체 클래스 검출 수
                carved_out = 0
                if hand_conf_map is not None and len(masks) > 0:
                    # robot_hand 우선 규칙(신뢰도 비교판): 손과 겹친 다른 클래스 픽셀은
                    # 손 검출 신뢰도가 그 물체 검출 신뢰도보다 높을 때만 손으로 본다.
                    # 무조건 빼면 물체 손잡이에 뜬 저신뢰 robot_hand 오검출이 진짜
                    # 손잡이를 지운다(O 실물 시험의 텀블러 0.32 사례). 기능부 오염은
                    # 회피 지도(functional_mask)를 부풀려 접근 중 후보를 부당하게
                    # 거부하게 만들므로 특히 해롭다.
                    veto = hand_conf_map[None, :, :] > confidences[:, None, None]
                    area_before = masks.sum(axis=(1, 2))
                    masks = masks & ~veto
                    area_after = masks.sum(axis=(1, 2))
                    # 절반 이상이 손 픽셀이던 검출은 파편을 남기지 말고 통째로 버린다.
                    mostly_hand = area_after < 0.5 * np.maximum(area_before, 1)
                    if mostly_hand.any():
                        carved_out = int(mostly_hand.sum())
                        keep = ~mostly_hand
                        masks, class_ids, confidences = masks[keep], class_ids[keep], confidences[keep]
                candidates, functional_mask = extract_candidates_from_arrays(
                    masks, class_ids, confidences, frame.shape,
                    min_confidence=selection_config["candidate"]["min_confidence"],
                    min_area_px=selection_config["candidate"]["min_area_px"],
                    boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
                )
                extracted = len(candidates)             # 후보 추출 통과 수
                suppressed = {"unstable": 0, "hand_overlap": carved_out}
                if suppression_filter is not None:
                    # 추적이 이번 프레임에 유효할 때만 박스를 쓴다(스테일 박스 금지).
                    if hand_box is None and hand_tracker is not None and hand_state is not None:
                        hand_box = hand_tracker.last_box
                    candidates, filter_suppressed = suppression_filter.update(candidates, hand_box=hand_box)
                    suppressed["unstable"] += filter_suppressed["unstable"]
                    suppressed["hand_overlap"] += filter_suppressed["hand_overlap"]
                    suppressed["ghost"] = filter_suppressed.get("ghost", 0)
                decision = decide_grasp(candidates, functional_mask, hand_state, selection_config)
                if suppression_filter is not None:
                    # 파지점 안정화: 가림 중에는 마지막 검증 파지점을 유지하고,
                    # 온전할 때는 EMA로 떨림을 제거한다 (자세한 근거는 hand_filter.py).
                    suppression_filter.stabilize_points(decision.ranked)
                draw_selection(overlay, decision, hand_state)
                # 파지점(하얀 점)이 사라진 이유를 화면에서 바로 읽을 수 있게, 후보가
                # 각 단계에서 몇 개 탈락했는지 사유별로 표시한다.
                rejected_functional = sum(
                    1 for c in candidates if c.scores.get("reject_reason") == "too_close_to_functional"
                )
                rejected_width = sum(
                    1 for c in candidates if c.scores.get("reject_reason") == "max_grasp_width_exceeded"
                )
                diagnosis = (
                    f"det={object_detections} cand={extracted}"
                    f" (conf/area -{object_detections - carved_out - extracted})"
                    f" unstable -{suppressed['unstable']} hand -{suppressed['hand_overlap']}"
                    f" hold +{suppressed.get('ghost', 0)}"
                    f" func -{rejected_functional} width -{rejected_width}"
                    f" ranked={len(decision.ranked)}"
                )
                cv2.putText(overlay, diagnosis, (10, overlay.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

            # 손 제어: 추적이 끊긴 프레임도 알려야 파지 직전 중단이 동작한다.
            if hand_controller is not None:
                hand_status = hand_controller.update(
                    decision, hand_valid=hand_state is not None
                )
                draw_hand_status(overlay, hand_status)

            fps_text = f"infer {elapsed_ms:.0f}ms ({1000.0 / elapsed_ms:.1f} FPS) conf>={config.conf_threshold}"
            cv2.putText(overlay, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if args.record:
                if writer is None:
                    video_path = output_dir / f"rfdetr_realtime_{stamp}.mp4"
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        max(2.0, min(15.0, 1000.0 / max(elapsed_ms, 66.0))),
                        (overlay.shape[1], overlay.shape[0]),
                    )
                writer.write(overlay)

            if not args.no_window:
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "user_quit"
                    break
                if hand_controller is not None:
                    if key == ord(" "):
                        hand_controller.request_close()
                    elif key == ord("o"):
                        hand_controller.request_open()
                    elif key == ord("x"):
                        hand_controller.emergency_stop()
                if key == ord("s"):
                    snap_path = output_dir / f"rfdetr_snapshot_{stamp}_{frames:04d}.png"
                    cv2.imwrite(str(snap_path), overlay)
                    print(f"스냅숏 저장: {snap_path}")

            if args.duration > 0 and (time.perf_counter() - start) >= args.duration:
                stop_reason = "duration_elapsed"
                break
    finally:
        # 어떤 경로로 끝나도 손을 먼저 세운다.
        if hand_controller is not None:
            try:
                hand_controller.emergency_stop()
            except Exception:
                pass
        if hand_client is not None:
            try:
                hand_client.shutdown()
            except Exception:
                pass
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    summary = {
        "model": str(model_path),
        "variant": args.variant,
        "camera_index": config.camera_index,
        "conf_threshold": config.conf_threshold,
        "frames": frames,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "mean_inference_ms": round(float(np.mean(inference_times)), 1) if inference_times else None,
        "median_inference_ms": round(float(np.median(inference_times)), 1) if inference_times else None,
        "effective_fps": round(1000.0 / float(np.mean(inference_times)), 2) if inference_times else None,
        "detections_total": total_counts,
        "device_requested": device,
        "device_actual": device_actual,
        "gpu_name": (
            torch.cuda.get_device_name(0) if device_actual.startswith("cuda") else None
        ),
        "class_id_base": class_id_base,
        "stop_reason": stop_reason,
    }
    if hand_controller is not None:
        summary["hand_state"] = hand_controller.status.state
        summary["hand_detail"] = hand_controller.status.detail
    summary_path = output_dir / f"rfdetr_realtime_{stamp}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
