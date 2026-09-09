#!/usr/bin/env python
"""부위 수준 파지 결정 실시간 시험판 (파지점 없는 변형 — A/B 비교용).

run_realtime_rfdetr.py(현행판)는 그대로 두고, 파지점 대신 "손 접근 × 물체 부위"로
자세를 결정하는 변형을 별도 실행 파일로 제공한다 (2026-08-26 사용자 제안).
결정 로직은 src/grasp_selection/part_scoring.py, 나머지 단계(검출, base 고정,
손 선택, 픽셀 억제, 억제 필터·유령 후보)는 현행판과 같은 구현을 공유한다.

표시: 흰 점 대신 **선택된 부위 마스크를 굵은 윤곽으로 강조**하고 자세를 표기한다.
--show-point를 주면 참고용으로 추출 단계의 내부점도 함께 그린다.

범위: 손 소스는 segmentation 고정, 마우스·배경차분·손 제어 연동은 미포함
(A/B 판정 후 채택되면 현행판에 통합한다).

사용 예 (젯슨):
  .venv/bin/python scripts/run_realtime_rfdetr_part.py \
      --model outputs/training/..._p_.../checkpoint_best_ema.pth --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_realtime_rfdetr import (  # noqa: E402
    CLASSES,
    ROBOT_HAND_CLASS,
    detections_to_arrays,
    render_overlay,
    resolve_repo_path,
)
from src.grasp_selection import HandSuppressionFilter, load_selection_config  # noqa: E402
from src.grasp_selection import extract_candidates_from_arrays  # noqa: E402
from src.grasp_selection.part_scoring import decide_grasp_by_part  # noqa: E402
from src.perception.hand_tracker import (  # noqa: E402
    HandMotionEstimator,
    SegmentationHandSelector,
    load_hand_tracker_config,
)
from src.perception.realtime import load_realtime_config  # noqa: E402


def draw_part_selection(overlay: np.ndarray, decision, hand, show_point: bool) -> None:
    """선택 부위 강조 표시: 굵은 윤곽 + 순위·자세 라벨 (파지점 없음)."""

    for rank, candidate in enumerate(decision.ranked):
        is_top = decision.state == "GRASP" and candidate is decision.candidate
        held = bool(candidate.scores.get("point_held"))
        if held and not is_top:
            # 기억(유령) 후보는 1위일 때만 표시한다. 하위 순위 유령까지 그리면
            # 물체를 옮긴 뒤 옛 자리에 흰 윤곽 잔상이 여러 개 남는다 (2026-08-26 보고).
            continue
        if is_top and not held:
            color, thickness = (255, 255, 255), 4          # 실측 선택: 굵은 흰색
        elif is_top:
            color, thickness = (150, 150, 150), 2          # 기억 유지 선택: 회색 가는 선
        else:
            color, thickness = (170, 170, 170), 2
        contours, _ = cv2.findContours(candidate.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness)
        if not contours:
            continue
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        label = f"#{rank + 1} {candidate.scores['total']:.2f}"
        if is_top:
            label += f" -> {decision.pose}"
        if held:
            label += " HOLD"
        cv2.putText(overlay, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2 if is_top else 1, cv2.LINE_AA)
        if show_point:
            px, py = candidate.grasp_point
            cv2.circle(overlay, (int(px), int(py)), 6, color, 1, cv2.LINE_AA)

    if hand is not None:
        hx, hy = int(hand.position[0]), int(hand.position[1])
        cv2.drawMarker(overlay, (hx, hy), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
        if hand.direction is not None:
            tip = (int(hx + hand.direction[0] * 55), int(hy + hand.direction[1] * 55))
            cv2.arrowedLine(overlay, (hx, hy), tip, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    banner = {"GRASP": f"GRASP: {decision.pose}", "ALIGN": "ALIGN (adjust hand)",
              "NO_TARGET": "NO TARGET"}[decision.state]
    banner_color = {"GRASP": (80, 200, 80), "ALIGN": (0, 200, 255),
                    "NO_TARGET": (120, 120, 120)}[decision.state]
    cv2.putText(overlay, banner, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                banner_color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="RFDETRSegNano")
    parser.add_argument("--config", default="configs/hardware/realtime_camera.yaml")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="자동 종료까지 초(0=q 키)")
    parser.add_argument("--record", action="store_true", help="overlay 영상을 녹화한다")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hand-seg-conf", type=float, default=0.5)
    parser.add_argument("--hand-tracker-config", default="configs/hardware/hand_tracker.yaml")
    parser.add_argument("--show-point", action="store_true", help="참고용 내부점도 표시")
    parser.add_argument("--hand-port", default=None,
                        help="Brunel Hand 시리얼 포트(예: /dev/ttyACM0). 주면 실제 손을 구동한다")
    parser.add_argument("--hand-mock", action="store_true",
                        help="실제 손 없이 모의 컨트롤러로 전체 흐름 점검")
    parser.add_argument("--hand-config", default="configs/hardware/hand_actuators.yaml")
    parser.add_argument("--hand-auto-close", action="store_true",
                        help="후보가 안정되면 트리거 없이 파지 (기본은 스페이스 키)")
    parser.add_argument("--hand-stable-frames", type=int, default=5)
    args = parser.parse_args()

    config = load_realtime_config(resolve_repo_path(args.config))
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.conf is not None:
        config.conf_threshold = args.conf
    output_dir = resolve_repo_path(str(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    import rfdetr

    model_path = resolve_repo_path(args.model)
    print(f"모델 로딩 중: {args.variant} ({model_path.name}), 장치={args.device}")
    model = getattr(rfdetr, args.variant)(pretrain_weights=str(model_path), device=args.device)

    selection_config = load_selection_config(
        resolve_repo_path("configs/grasp_selection.yaml"),
        resolve_repo_path("configs/human_grasp_prior.yaml"),
    )
    tracker_config = load_hand_tracker_config(resolve_repo_path(args.hand_tracker_config))
    hand_estimator = HandMotionEstimator(
        ema_alpha=tracker_config.direction_ema_alpha,
        min_speed_px=tracker_config.min_speed_px,
        hand_width_mm=tracker_config.hand_width_mm,
        width_px_decay=tracker_config.width_px_decay,
    )
    hand_selector = SegmentationHandSelector(miss_ttl_frames=tracker_config.max_missing_frames)
    suppression_filter = HandSuppressionFilter.from_config(selection_config)

    # 손 제어 연동 (현행판과 같은 구성: 스페이스=파지, o=펼침, x=비상정지)
    hand_client = None
    hand_controller = None
    draw_hand_status = None
    if args.hand_port or args.hand_mock:
        from datetime import datetime

        from src.hand_control import (
            GraspController,
            HandClient,
            MockLink,
            SerialLink,
            load_calibration,
        )
        from src.perception.realtime import draw_hand_status  # noqa: F811

        hand_calibration = load_calibration(resolve_repo_path(args.hand_config))
        hand_log_dir = REPOSITORY_ROOT / "outputs" / "hand_tests"
        hand_log_dir.mkdir(parents=True, exist_ok=True)
        hand_log_path = hand_log_dir / f"part_hand_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
            hand_client, hand_calibration,
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

    window_name = "rfdetr part mode"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("부위 수준 결정 모드: 파지점 없음, 선택 부위 강조 표시")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    writer = None
    class_id_base = 0
    class_id_base_locked = False
    hand_state = None
    hand_missing_streak = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            t0 = time.perf_counter()
            detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                       threshold=config.conf_threshold)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if detections is not None and len(detections) > 0 and not class_id_base_locked:
                observed = np.asarray(detections.class_id, dtype=int)
                if int(observed.min()) == 0:
                    class_id_base, class_id_base_locked = 0, True
                elif int(observed.max()) > ROBOT_HAND_CLASS:
                    class_id_base, class_id_base_locked = 1, True

            overlay, _ = render_overlay(frame, detections, class_id_base, config.overlay_alpha,
                                        robot_hand_min_conf=args.hand_seg_conf)

            masks_all, ids_all, confs_all = detections_to_arrays(detections, class_id_base)
            hand_select = ids_all == ROBOT_HAND_CLASS
            hand_conf_map = None
            if hand_select.any():
                hand_conf_map = np.zeros(masks_all.shape[1:], dtype=np.float32)
                for hand_mask, hand_conf in zip(masks_all[hand_select], confs_all[hand_select]):
                    np.maximum(hand_conf_map, hand_mask * float(hand_conf), out=hand_conf_map)
            track_ids = np.flatnonzero(hand_select & (confs_all >= args.hand_seg_conf))
            hand_infos = []
            for i in track_ids:
                ys, xs = np.nonzero(masks_all[i])
                if len(xs):
                    hand_infos.append((int(i), (float(xs.mean()), float(ys.mean())),
                                       (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
            pick = hand_selector.select([h[1] for h in hand_infos],
                                        [float(confs_all[h[0]]) for h in hand_infos])
            hand_box = None
            if pick is not None:
                chosen = hand_infos[pick]
                x1, y1, x2, y2 = chosen[2]
                merge_margin = 0.6 * max(x2 - x1 + 1, y2 - y1 + 1)
                union = masks_all[chosen[0]].copy()
                for info in hand_infos:
                    if info is not chosen:
                        cx, cy = info[1]
                        if (x1 - merge_margin <= cx <= x2 + merge_margin
                                and y1 - merge_margin <= cy <= y2 + merge_margin):
                            union |= masks_all[info[0]]
                ys, xs = np.nonzero(union)
                hand_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                box_short = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
                hand_state = hand_estimator.observe((float(xs.mean()), float(ys.mean())), box_short)
                hand_missing_streak = 0
                cv2.rectangle(overlay, hand_box[:2], hand_box[2:], (0, 255, 255), 2)
                cv2.putText(overlay, "robot hand", (hand_box[0], max(18, hand_box[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                hand_missing_streak += 1
                if hand_missing_streak > tracker_config.max_missing_frames:
                    hand_state = None
                    hand_estimator.reset()
                if hand_state is None:
                    cv2.putText(overlay, "HAND NOT DETECTED", (10, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            decision = None
            if hand_state is not None:
                masks = masks_all[~hand_select]
                class_ids = ids_all[~hand_select]
                confidences = confs_all[~hand_select]
                object_detections = len(masks)
                carved_out = 0
                if hand_conf_map is not None and len(masks) > 0:
                    veto = hand_conf_map[None, :, :] > confidences[:, None, None]
                    area_before = masks.sum(axis=(1, 2))
                    masks = masks & ~veto
                    area_after = masks.sum(axis=(1, 2))
                    mostly_hand = area_after < 0.5 * np.maximum(area_before, 1)
                    if mostly_hand.any():
                        carved_out = int(mostly_hand.sum())
                        keep = ~mostly_hand
                        masks, class_ids, confidences = (
                            masks[keep], class_ids[keep], confidences[keep])
                candidates, functional_mask = extract_candidates_from_arrays(
                    masks, class_ids, confidences, frame.shape,
                    min_confidence=selection_config["candidate"]["min_confidence"],
                    min_area_px=selection_config["candidate"]["min_area_px"],
                    boundary_margin_px=selection_config["candidate"]["boundary_margin_px"],
                )
                extracted = len(candidates)
                candidates, suppressed = suppression_filter.update(candidates, hand_box=hand_box)
                suppressed["hand_overlap"] += carved_out
                decision = decide_grasp_by_part(candidates, functional_mask, hand_state,
                                                selection_config)
                suppression_filter.stabilize_points(decision.ranked)   # 유령 스냅샷 유지용
                draw_part_selection(overlay, decision, hand_state, args.show_point)
                rejected_entangled = sum(
                    1 for c in candidates
                    if c.scores.get("reject_reason") == "functional_entangled")
                diagnosis = (
                    f"det={object_detections} cand={extracted}"
                    f" unstable -{suppressed['unstable']} hand -{suppressed['hand_overlap']}"
                    f" hold +{suppressed.get('ghost', 0)} entangle -{rejected_entangled}"
                    f" ranked={len(decision.ranked)}"
                )
                cv2.putText(overlay, diagnosis, (10, overlay.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

            # 손 제어: 추적이 끊긴 프레임도 알려야 파지 직전 중단이 동작한다.
            if hand_controller is not None:
                hand_status = hand_controller.update(decision, hand_valid=hand_state is not None)
                draw_hand_status(overlay, hand_status)

            fps_text = (f"infer {elapsed_ms:.0f}ms ({1000.0 / max(elapsed_ms, 1.0):.1f} FPS) "
                        f"conf>={config.conf_threshold}  [PART]")
            cv2.putText(overlay, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if args.record:
                if writer is None:
                    video_path = output_dir / f"rfdetr_part_{stamp}.mp4"
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
                    break
                if hand_controller is not None:
                    if key == ord(" "):
                        hand_controller.request_close()
                    elif key == ord("o"):
                        hand_controller.request_open()
                    elif key == ord("x"):
                        hand_controller.emergency_stop()
            if args.duration > 0 and time.perf_counter() - start >= args.duration:
                break
    finally:
        # 어떤 경로로 끝나도 손을 먼저 세운다 (현행판과 동일).
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
    print("종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
