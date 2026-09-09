#!/usr/bin/env python
"""Brunel Hand 펌웨어용 시리얼 터미널 + 로그 기록기.

Arduino IDE 시리얼 모니터는 스크롤백이 제한되고 전체 복사가 잘 안 되므로,
이 스크립트로 명령을 보내고 모든 수신 줄을 타임스탬프와 함께 파일에 남긴다.
보정 측정(끝 위치 ADC, 잡음 폭, 방향 확인) 기록에 쓴다.

- 키보드로 입력한 줄을 그대로 보낸다(줄 끝에 LF 추가). 현재 펄스 펌웨어는
  `1`~`8`, `s`, `p` 단일 문자 명령이므로 한 글자씩 입력 후 Enter.
- 수신 줄은 화면과 로그 파일에 동시에 기록한다.
- 종료(Ctrl+C 또는 `quit`) 시 정지 명령 `s`를 한 번 보낸다(--no-stop-on-exit로 끔).
- 이 스크립트가 포트를 잡는 동안 Arduino IDE 시리얼 모니터는 닫아 둬야 한다.

사용 예:
    python scripts/hand_serial_terminal.py --list
    python scripts/hand_serial_terminal.py --port COM5
    python scripts/hand_serial_terminal.py --port COM5 --log outputs/hand_tests/index_calib.log

의존성: pyserial (pip install pyserial)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import threading
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - 의존성 안내용
    print("pyserial이 없습니다. 먼저 설치하세요:  pip install pyserial", file=sys.stderr)
    raise SystemExit(1)


def print_ports() -> None:
    """연결된 시리얼 포트 목록을 출력한다."""

    ports = list(list_ports.comports())
    if not ports:
        print("시리얼 포트를 찾지 못했습니다. USB 연결을 확인하세요.")
        return
    for port in ports:
        print(f"{port.device:10s} {port.description}  [{port.hwid}]")


def reader_loop(link: serial.Serial, log_handle, stop_event: threading.Event) -> None:
    """수신 줄을 화면과 로그 파일에 기록한다."""

    while not stop_event.is_set():
        try:
            raw = link.readline()
        except serial.SerialException as error:
            print(f"\n[수신 오류] {error}", file=sys.stderr)
            stop_event.set()
            break
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{stamp}] {text}")
        log_handle.write(f"{stamp} RX {text}\n")
        log_handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="시리얼 포트 목록만 출력하고 종료")
    parser.add_argument("--port", default=None, help="시리얼 포트 (예: COM5, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="통신 속도(기본 115200)")
    parser.add_argument(
        "--log",
        default=None,
        help="로그 파일 경로(기본: outputs/hand_tests/serial_<날짜시각>.log)",
    )
    parser.add_argument("--no-stop-on-exit", action="store_true", help="종료 시 정지 명령 s를 보내지 않음")
    args = parser.parse_args()

    if args.list:
        print_ports()
        return 0
    if not args.port:
        print("--port를 지정하세요. 포트 목록: --list", file=sys.stderr)
        print_ports()
        return 2

    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = REPOSITORY_ROOT / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = REPOSITORY_ROOT / "outputs" / "hand_tests" / f"serial_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        link = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        print(f"포트를 열 수 없습니다: {error}\n(Arduino IDE 시리얼 모니터가 열려 있으면 닫으세요)", file=sys.stderr)
        return 1

    print(f"연결: {args.port} @ {args.baud}  로그: {log_path}")
    print("명령을 입력하고 Enter. 종료: quit 또는 Ctrl+C. (현재 펌웨어: 1~8 이동, s 정지, p 위치)")

    stop_event = threading.Event()
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"# {datetime.now().isoformat(timespec='seconds')} 연결 {args.port} @ {args.baud}\n")
        log_handle.flush()
        reader = threading.Thread(target=reader_loop, args=(link, log_handle, stop_event), daemon=True)
        reader.start()
        try:
            while not stop_event.is_set():
                try:
                    line = input()
                except EOFError:
                    break
                command = line.strip()
                if command.lower() in {"quit", "exit"}:
                    break
                if not command:
                    continue
                link.write((command + "\n").encode("utf-8"))
                stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log_handle.write(f"{stamp} TX {command}\n")
                log_handle.flush()
        except KeyboardInterrupt:
            pass
        finally:
            if not args.no_stop_on_exit:
                try:
                    link.write(b"s\n")
                    log_handle.write(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} TX s (종료 시 정지)\n")
                    time.sleep(0.3)  # 정지 응답을 받아 기록할 시간
                except serial.SerialException:
                    pass
            stop_event.set()
            reader.join(timeout=1.0)
            link.close()
            log_handle.write(f"# {datetime.now().isoformat(timespec='seconds')} 종료\n")
    print(f"로그 저장: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
