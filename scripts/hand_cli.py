#!/usr/bin/env python
"""Brunel Hand 제어 CLI (젯슨·PC 공용).

`configs/hardware/hand_actuators.yaml`의 보정값으로 펌웨어에 연결하고,
읽기 전용 점검부터 프리셋 구동까지 한 명령씩 수행한다. 모든 송수신은
`outputs/hand_tests/`에 로그로 남는다.

예시:
    python scripts/hand_cli.py selftest                      # 모터 구동 없음
    python scripts/hand_cli.py pos
    python scripts/hand_cli.py frac 0 0.5                    # 검지 반 접기
    python scripts/hand_cli.py fracall 1.0 1.0 1.0 -         # 엄지 제외 접기
    python scripts/hand_cli.py open
    python scripts/hand_cli.py demo                          # 안전 순서 시연
    python scripts/hand_cli.py --mock selftest               # 하드웨어 없이 흐름 점검

포트는 --port로 주거나 YAML의 serial.port를 쓴다. 젯슨에서는 udev 규칙을 넣으면
`/dev/brunel_hand`로 고정할 수 있다(docs/jetson_hand_control_plan.md 참고).

안전: 10 V 모터 전원 인가는 사람이 한다. 전원이 꺼져 있으면 이동 명령은
`EVT STALL`로 끝나며 손은 움직이지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.hand_control import (  # noqa: E402
    CalibrationError,
    HandClient,
    HandError,
    LinkError,
    MockLink,
    SerialLink,
    load_calibration,
)
from src.hand_control import protocol  # noqa: E402

DEFAULT_CONFIG = "configs/hardware/hand_actuators.yaml"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def parse_fraction(token: str) -> float | None:
    """'-'는 '그대로 두기', 나머지는 0.0~1.0 비율."""

    if token in {"-", "none", "None"}:
        return None
    value = float(token)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"비율은 0.0~1.0이어야 합니다: {token}")
    return value


def describe_positions(calibration, positions: list[int]) -> str:
    parts = []
    for finger, adc in zip(calibration.fingers, positions):
        fraction = finger.adc_to_fraction(adc)
        # int()로 감싸 -0.0이 "-0%"로 보이는 것을 막는다
        percent = int(round(fraction * 100))
        parts.append(f"{protocol.finger_label(finger.index)} {adc}({percent}%접힘)")
    return " | ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="보정 설정 YAML")
    parser.add_argument("--port", default=None, help="시리얼 포트(기본: YAML serial.port)")
    parser.add_argument("--baud", type=int, default=None, help="통신 속도(기본: YAML)")
    parser.add_argument("--mock", action="store_true", help="하드웨어 없이 모의 연결로 실행")
    parser.add_argument("--log", default=None, help="로그 파일 경로(기본: outputs/hand_tests 자동 이름)")
    parser.add_argument("--timeout", type=float, default=8.0, help="이동 완료 대기 상한(초)")
    parser.add_argument("--quiet", action="store_true", help="송수신 줄을 화면에 찍지 않는다")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="연결·펌웨어·보정 확인 (모터 구동 없음)")
    sub.add_parser("pos", help="현재 위치값 출력")
    sub.add_parser("status", help="손가락별 상태 출력")
    sub.add_parser("stop", help="즉시 전체 정지")
    sub.add_parser("params", help="펌웨어 설정값 출력")

    open_parser = sub.add_parser("open", help="펼치기")
    open_parser.add_argument("finger", nargs="?", type=int, help="손가락 번호(생략하면 전체)")

    close_parser = sub.add_parser("close", help="접기")
    close_parser.add_argument("finger", nargs="?", type=int, help="손가락 번호(생략하면 전체)")

    frac_parser = sub.add_parser("frac", help="한 손가락을 닫힘 비율로 이동")
    frac_parser.add_argument("finger", type=int)
    frac_parser.add_argument("fraction", type=float, help="0.0(펼침)~1.0(접힘)")

    fracall_parser = sub.add_parser("fracall", help="네 손가락 동시 이동('-'는 유지)")
    fracall_parser.add_argument("fractions", nargs=4, type=parse_fraction)

    move_parser = sub.add_parser("move", help="한 손가락을 ADC 목표로 이동")
    move_parser.add_argument("finger", type=int)
    move_parser.add_argument("adc", type=int)

    preset_parser = sub.add_parser("preset", help="YAML 프리셋으로 이동")
    preset_parser.add_argument("name", help="OPEN/PRECISION/WRAP/POWER 등")

    raw_parser = sub.add_parser("raw", help="펌웨어 명령을 그대로 보낸다")
    raw_parser.add_argument("text", help='예: "MOVE 0 3000"')

    sub.add_parser("demo", help="안전 순서 시연(작은 이동 → 복귀)")
    sub.add_parser("presets", help="사용 가능한 프리셋 목록")
    return parser


def run_command(client: HandClient, calibration, args) -> dict:
    """부명령을 실행하고 결과 요약을 돌려준다."""

    command = args.command

    if command == "selftest":
        identity = client.identity or {}
        positions = client.positions()
        params = client.params()
        return {
            "command": command,
            "identity": identity,
            "positions": positions,
            "positions_text": describe_positions(calibration, positions),
            "params": params,
            "note": "모터를 구동하지 않았습니다.",
        }

    if command == "pos":
        positions = client.positions()
        return {"command": command, "positions": positions,
                "positions_text": describe_positions(calibration, positions)}

    if command == "status":
        entries = client.status()
        return {"command": command, "status": [
            {"finger": e.index, "label": protocol.finger_label(e.index),
             "state": e.state, "adc": e.adc, "target": e.target} for e in entries]}

    if command == "params":
        return {"command": command, "params": client.params()}

    if command == "stop":
        client.stop()
        return {"command": command, "result": "정지"}

    if command == "presets":
        return {"command": command, "available": calibration.available_presets(),
                "undefined": sorted(n for n, v in calibration.presets.items() if v is None)}

    if command == "raw":
        response, events = client.send_raw(args.text, collect_events=1.0)
        return {"command": command, "sent": args.text, "response": response.raw,
                "events": [event.raw for event in events]}

    # ---- 이동 명령들 ----
    if command == "open":
        client.open(args.finger)
        waiting = None if args.finger is None else {args.finger}
    elif command == "close":
        client.close_hand(args.finger)
        waiting = None if args.finger is None else {args.finger}
    elif command == "frac":
        client.move_fraction(args.finger, args.fraction)
        waiting = {args.finger}
    elif command == "fracall":
        client.move_fractions(list(args.fractions))
        waiting = {i for i, value in enumerate(args.fractions) if value is not None}
    elif command == "move":
        client.move_adc(args.finger, args.adc)
        waiting = {args.finger}
    elif command == "preset":
        # 프리셋은 정의된 순서(엄지 먼저 등)대로 단계별로 구동한다.
        results = client.run_preset(args.name, timeout=args.timeout)
        positions = client.positions()
        return {
            "command": command,
            "preset": args.name.upper(),
            "steps": [list(step) for step in calibration.preset_steps(args.name)],
            "results": {
                str(index): {"label": protocol.finger_label(index), "kind": result.kind,
                             "adc": result.adc}
                for index, result in sorted(results.items())
            },
            "positions": positions,
            "positions_text": describe_positions(calibration, positions),
        }
    elif command == "demo":
        return run_demo(client, calibration, args)
    else:  # pragma: no cover
        raise HandError(f"모르는 명령: {command}")

    results = client.wait_settled(waiting, timeout=args.timeout)
    positions = client.positions()
    return {
        "command": command,
        "results": {
            str(index): {"label": protocol.finger_label(index), "kind": result.kind, "adc": result.adc}
            for index, result in sorted(results.items())
        },
        "positions": positions,
        "positions_text": describe_positions(calibration, positions),
    }


def run_demo(client: HandClient, calibration, args) -> dict:
    """작은 이동과 복귀만 하는 안전 시연.

    각 손가락을 현재 위치에서 10 %만 접었다 펴며, 결과를 순서대로 기록한다.
    전원이 꺼져 있으면 모두 STALL로 끝나므로 배선·전원 점검에도 쓸 수 있다.
    """

    steps = []
    start = client.positions()
    for finger in calibration.fingers:
        current = finger.adc_to_fraction(start[finger.index])
        target = min(1.0, max(0.0, current) + 0.10)
        client.move_fraction(finger.index, target)
        moved = client.wait_settled({finger.index}, timeout=args.timeout)
        steps.append({
            "finger": finger.index,
            "label": protocol.finger_label(finger.index),
            "from_adc": start[finger.index],
            "target_fraction": round(target, 3),
            "kind": moved[finger.index].kind,
            "adc": moved[finger.index].adc,
        })
        client.move_fraction(finger.index, max(0.0, current))
        client.wait_settled({finger.index}, timeout=args.timeout)
    positions = client.positions()
    return {
        "command": "demo",
        "steps": steps,
        "positions": positions,
        "positions_text": describe_positions(calibration, positions),
    }


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)

    try:
        calibration = load_calibration(config_path)
    except CalibrationError as error:
        print(f"보정 설정 오류: {error}", file=sys.stderr)
        return 2

    if args.log:
        log_path = resolve_path(args.log)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = REPOSITORY_ROOT / "outputs" / "hand_tests" / f"hand_cli_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    def record(text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_handle.write(f"{stamp} {text}\n")
        log_handle.flush()
        if not args.quiet:
            print(f"[{stamp}] {text}")

    record(f"# 명령={args.command} 설정={config_path.name} mock={args.mock}")

    try:
        if args.mock:
            link = MockLink(calibration)
        else:
            port = args.port or calibration.port
            if not port:
                print("포트를 지정하세요(--port 또는 YAML serial.port).", file=sys.stderr)
                return 2
            link = SerialLink(port, args.baud or calibration.baud)
            record(f"# 연결 {link.description}")
    except LinkError as error:
        print(f"연결 실패: {error}", file=sys.stderr)
        return 1

    client = HandClient(link, calibration, event_log=record)
    exit_code = 0
    try:
        client.connect()
        summary = run_command(client, calibration, args)
        summary["log"] = str(log_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (HandError, LinkError, CalibrationError) as error:
        print(f"실패: {error}", file=sys.stderr)
        record(f"# 실패 {error}")
        exit_code = 1
    finally:
        client.shutdown()
        record("# 종료")
        log_handle.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
