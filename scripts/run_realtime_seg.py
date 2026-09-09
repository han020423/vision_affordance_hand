#!/usr/bin/env python
"""학습된 YOLO11n-seg 가중치로 실시간 affordance 분할을 실행한다.

기본값은 configs/hardware/realtime_camera.yaml에서 읽고 CLI 인자가 이를
덮어쓴다. --image를 주면 카메라 없이 단일 이미지로 추론과 시각화를 검증한다.

예시:
    python scripts/run_realtime_seg.py \
      --model outputs/training/custom_finetune_d_public_seed42/weights/best.pt

조작 키: q/ESC 종료, s 스냅숏 저장
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.perception.realtime import (  # noqa: E402
    infer_single_image,
    load_realtime_config,
    run_camera_loop,
)


def resolve_repo_path(value: str) -> Path:
    """저장소 상대경로를 절대경로로 해석한다."""

    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="outputs/training/custom_finetune_d_public_seed42/weights/best.pt",
        help="학습된 세그멘테이션 가중치 경로",
    )
    parser.add_argument(
        "--config",
        default="configs/hardware/realtime_camera.yaml",
        help="카메라·추론 기본 설정 YAML",
    )
    parser.add_argument("--camera-index", type=int, default=None, help="카메라 장치 번호(설정값 덮어쓰기)")
    parser.add_argument("--width", type=int, default=None, help="카메라 가로 해상도")
    parser.add_argument("--height", type=int, default=None, help="카메라 세로 해상도")
    parser.add_argument("--imgsz", type=int, default=None, help="추론 입력 크기")
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold")
    parser.add_argument("--device", default=None, help="추론 장치 (cpu, cuda:0 등)")
    parser.add_argument("--record", action="store_true", help="overlay 영상을 녹화한다")
    parser.add_argument("--video", default=None, help="카메라 대신 이 영상 파일로 오프라인 검증한다")
    parser.add_argument("--no-display", action="store_true", help="창을 띄우지 않는다 (녹화·배치용)")
    parser.add_argument("--max-frames", type=int, default=None, help="이 프레임 수만 처리하고 종료 (배치 검증용)")
    parser.add_argument(
        "--image",
        default=None,
        help="카메라 대신 이 이미지 한 장으로 추론을 검증하고 overlay를 저장한다",
    )
    parser.add_argument(
        "--image-output",
        default=None,
        help="--image 모드의 overlay 저장 경로(기본: 출력 폴더 아래 자동 이름)",
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
    parser.add_argument("--hand-x", type=float, default=None, help="--image 모드 모의 손 x")
    parser.add_argument("--hand-y", type=float, default=None, help="--image 모드 모의 손 y")
    parser.add_argument(
        "--hand-source",
        choices=["mouse", "detector"],
        default="mouse",
        help="후보 선택의 손 위치 소스: mouse(모의 손) 또는 detector(로봇손 비전 검출)",
    )
    parser.add_argument(
        "--hand-tracker-config",
        default="configs/hardware/hand_tracker.yaml",
        help="로봇손 추적 설정 YAML",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Brunel Hand 컨트롤러 시리얼 포트(예: COM10, /dev/brunel_hand). 주면 실제 손을 구동한다",
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
    # CLI 인자가 있으면 YAML 기본값을 덮어쓴다.
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.width is not None:
        config.width = args.width
    if args.height is not None:
        config.height = args.height
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.conf is not None:
        config.conf_threshold = args.conf
    if args.device is not None:
        config.device = args.device
    if args.record:
        config.record_video = True
    if args.video:
        config.video_path = resolve_repo_path(args.video)
    if args.no_display:
        config.display = False
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    config.output_dir = resolve_repo_path(str(config.output_dir))

    model_path = resolve_repo_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 가중치가 없습니다: {model_path}")

    # Ultralytics를 불러오기 전에 저장소 내부 설정 경로를 지정해 사용자 홈에
    # settings.json을 쓰려는 동작과 권한 경고를 막는다(train_seg.py와 동일).
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPOSITORY_ROOT / ".ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    selection_config = None
    if args.select:
        from src.grasp_selection import load_selection_config

        selection_config = load_selection_config(
            resolve_repo_path(args.selection_config),
            resolve_repo_path(args.human_prior_config),
        )

    if args.image is not None:
        image_path = resolve_repo_path(args.image)
        if args.image_output is not None:
            output_path = resolve_repo_path(args.image_output)
        else:
            output_path = config.output_dir / f"single_image_overlay_{image_path.stem}.png"
        if selection_config is not None:
            import cv2

            from src.grasp_selection import HandState, decide_grasp, extract_candidates
            from src.perception.realtime import draw_selection, render_affordance_overlay

            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
            result = model.predict(frame, imgsz=config.imgsz, conf=config.conf_threshold,
                                   device=config.device, verbose=False)[0]
            overlay, detections = render_affordance_overlay(frame, result, config.overlay_alpha)
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
            candidates, functional_mask = extract_candidates(
                result, frame.shape,
                min_confidence=selection_config["candidate"]["min_confidence"],
                min_area_px=selection_config["candidate"]["min_area_px"],
                boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
            )
            decision = decide_grasp(candidates, functional_mask, hand, selection_config)
            draw_selection(overlay, decision, hand)
            output_path.parent.mkdir(parents=True, exist_ok=True)
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
        detections = infer_single_image(model, image_path, config, output_path)
        print(
            json.dumps(
                {
                    "mode": "single_image",
                    "model": str(model_path),
                    "image": str(image_path),
                    "overlay_saved": str(output_path),
                    "detections": detections,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(
        f"실시간 추론 시작: 모델={model_path.name}, 카메라={config.camera_index}, "
        f"해상도={config.width}x{config.height}, device={config.device}"
    )
    print("조작: q 또는 ESC 종료, s 스냅숏 저장")
    hand_tracker = None
    if selection_config is not None and args.hand_source == "detector":
        from src.perception.hand_tracker import HandTracker, load_hand_tracker_config

        tracker_config = load_hand_tracker_config(resolve_repo_path(args.hand_tracker_config))
        hand_tracker = HandTracker(tracker_config, REPOSITORY_ROOT)
        print(f"후보 선택 모드: 로봇손 비전 검출 ({tracker_config.backend}, 매 {tracker_config.detect_every}프레임)")
    elif selection_config is not None:
        print("후보 선택 모드: 마우스 커서 = 모의 손 위치, 커서 이동 방향 = 접근 방향")
    hand_client = None
    hand_controller = None
    if args.hand_port or args.hand_mock:
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

        hand_calibration = load_calibration(resolve_repo_path(args.hand_config))
        hand_log_dir = REPOSITORY_ROOT / "outputs" / "hand_tests"
        hand_log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hand_log_path = hand_log_dir / f"realtime_hand_{stamp}.log"
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
        if not hand_calibration.available_presets() or "WRAP" not in hand_calibration.available_presets():
            print(
                "주의: PRECISION/WRAP/POWER 프리셋이 아직 정해지지 않았습니다. "
                "파지 단계에서 구동을 거부하고 상태 표시만 합니다.",
                file=sys.stderr,
            )

    try:
        summary = run_camera_loop(model, config, selection_config, hand_tracker, hand_controller)
    finally:
        if hand_client is not None:
            hand_client.shutdown()
    summary["model"] = str(model_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["stop_reason"] == "camera_disconnected":
        print("카메라 연결이 끊겨 안전하게 종료했습니다. 장치를 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
