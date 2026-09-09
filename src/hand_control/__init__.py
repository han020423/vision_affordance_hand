"""Brunel Hand 저수준 제어 패키지 (호스트 쪽).

젯슨이나 PC에서 USB 시리얼로 `firmware/hand_closed_loop` 펌웨어를 구동한다.
보정값·프리셋의 단일 진실은 `configs/hardware/hand_actuators.yaml`이고,
펌웨어에는 연결 시 주입한다.

    from src.hand_control import HandClient, SerialLink, load_calibration

    calibration = load_calibration(Path("configs/hardware/hand_actuators.yaml"))
    with HandClient(SerialLink("/dev/brunel_hand"), calibration) as hand:
        hand.open()
        hand.wait_settled()
"""

from .client import HandClient, HandError, MoveResult, build_serial_client
from .link import LinkError, MockLink, SerialLink
from .presets import (
    CalibrationError,
    FingerCalibration,
    HandCalibration,
    load_calibration,
)
from .protocol import Event, FingerStatus, Response, finger_label, parse_line
from .state_machine import ControllerStatus, GraspController

__all__ = [
    "HandClient",
    "HandError",
    "MoveResult",
    "build_serial_client",
    "SerialLink",
    "MockLink",
    "LinkError",
    "load_calibration",
    "HandCalibration",
    "FingerCalibration",
    "CalibrationError",
    "Response",
    "Event",
    "FingerStatus",
    "parse_line",
    "finger_label",
    "GraspController",
    "ControllerStatus",
]
