"""Brunel Hand 펌웨어 시리얼 프로토콜의 인코딩과 파싱.

`firmware/hand_closed_loop`와 주고받는 줄 단위 ASCII를 다룬다. 이 모듈은
입출력을 하지 않으므로 하드웨어 없이 단위 시험할 수 있다.

수신 줄은 세 종류다.

- `OK <종류> [인자...]`  : 명령 성공 응답
- `ERR <코드> [설명]`     : 명령 실패
- `EVT <종류> [...]`      : 비동기 이벤트(이동 종료, 스톨 등)
"""

from __future__ import annotations

from dataclasses import dataclass

# 펌웨어가 `ID`에 응답하는 이름과 프로토콜 번호. 다르면 연결을 거부한다.
FIRMWARE_NAME = "hand_closed_loop"
PROTOCOL_VERSION = 1

# 이동이 끝났음을 뜻하는 이벤트(성공·실패 모두 포함).
SETTLE_EVENTS = frozenset({"DONE", "STALL", "TIMEOUT", "FAULT"})

FINGER_NAMES = ("index", "middle", "ring_little", "thumb")
FINGER_LABELS_KO = ("검지", "중지", "약지·소지", "엄지")


@dataclass(frozen=True)
class Response:
    """`OK ...` 또는 `ERR ...` 한 줄."""

    ok: bool
    kind: str                  # OK면 종류 토큰(ID/POS/MOVE...), ERR면 오류 코드
    args: tuple[str, ...]
    raw: str

    @property
    def failed(self) -> bool:
        return not self.ok


@dataclass(frozen=True)
class Event:
    """`EVT ...` 한 줄."""

    kind: str
    finger: int | None
    adc: int | None
    raw: str

    @property
    def is_settle(self) -> bool:
        """이동 종료 이벤트인지."""

        return self.kind in SETTLE_EVENTS


@dataclass(frozen=True)
class FingerStatus:
    """`STATUS` 응답의 손가락 하나."""

    index: int
    state: str
    adc: int
    target: int | None


def parse_line(line: str) -> Response | Event | None:
    """수신 줄 하나를 Response 또는 Event로 해석한다.

    빈 줄, 주석(`#`), 형식을 알 수 없는 줄은 None을 돌려준다.
    """

    text = line.strip()
    if not text or text.startswith("#"):
        return None
    tokens = text.split()
    head = tokens[0]
    if head == "OK":
        kind = tokens[1] if len(tokens) > 1 else ""
        return Response(ok=True, kind=kind, args=tuple(tokens[2:]), raw=text)
    if head == "ERR":
        kind = tokens[1] if len(tokens) > 1 else ""
        return Response(ok=False, kind=kind, args=tuple(tokens[2:]), raw=text)
    if head == "EVT":
        kind = tokens[1] if len(tokens) > 1 else ""
        finger = _maybe_int(tokens[2]) if len(tokens) > 2 else None
        adc = _maybe_int(tokens[3]) if len(tokens) > 3 else None
        return Event(kind=kind, finger=finger, adc=adc, raw=text)
    return None


def _maybe_int(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


def parse_identity(response: Response) -> dict[str, str]:
    """`OK ID <이름> fw <n> proto <n> fingers <n> ...`을 딕셔너리로 만든다.

    첫 인자는 펌웨어 이름이고 그 뒤는 `키 값` 쌍이다.
    """

    if not response.ok or response.kind != "ID":
        raise ValueError(f"ID 응답이 아닙니다: {response.raw}")
    if not response.args:
        raise ValueError(f"ID 응답에 펌웨어 이름이 없습니다: {response.raw}")
    identity = {"name": response.args[0]}
    rest = response.args[1:]
    for key, value in zip(rest[0::2], rest[1::2]):
        identity[key] = value
    return identity


def parse_positions(response: Response) -> list[int]:
    """`OK POS a0 a1 a2 a3`를 정수 목록으로 만든다."""

    if not response.ok or response.kind != "POS":
        raise ValueError(f"POS 응답이 아닙니다: {response.raw}")
    return [int(token) for token in response.args]


def parse_status(response: Response) -> list[FingerStatus]:
    """`OK STATUS i:STATE:adc:target ...`을 FingerStatus 목록으로 만든다."""

    if not response.ok or response.kind != "STATUS":
        raise ValueError(f"STATUS 응답이 아닙니다: {response.raw}")
    entries: list[FingerStatus] = []
    for token in response.args:
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(f"STATUS 항목 형식이 다릅니다: {token}")
        index, state, adc, target = parts
        entries.append(
            FingerStatus(
                index=int(index),
                state=state,
                adc=int(adc),
                target=None if target == "-" else int(target),
            )
        )
    return entries


def parse_calibration_echo(response: Response) -> dict[int, tuple[int, int, int, bool, int]]:
    """`GET CAL` 응답을 손가락별 (low, high, incdir, exthigh, maxmove)로 만든다.

    보정값이 없는 손가락은 `i:UNCAL`로 오므로 결과에서 제외한다.
    """

    if not response.ok or response.kind != "CAL":
        raise ValueError(f"CAL 응답이 아닙니다: {response.raw}")
    result: dict[int, tuple[int, int, int, bool, int]] = {}
    for token in response.args:
        parts = token.split(":")
        if len(parts) == 2 and parts[1] == "UNCAL":
            continue
        if len(parts) != 6:
            raise ValueError(f"CAL 항목 형식이 다릅니다: {token}")
        index, low, high, increase_direction, extended_is_high, max_move = parts
        result[int(index)] = (
            int(low),
            int(high),
            int(increase_direction),
            extended_is_high == "1",
            int(max_move),
        )
    return result


# ------------------------------------------------------------------ 명령 문자열


def command_identify() -> str:
    return "ID"


def command_ping() -> str:
    return "PING"


def command_stop() -> str:
    return "STOP"


def command_positions() -> str:
    return "POS"


def command_status() -> str:
    return "STATUS"


def command_move(finger: int, adc: int) -> str:
    return f"MOVE {int(finger)} {int(adc)}"


def command_move_all(targets: list[int | None]) -> str:
    values = " ".join("-1" if value is None else str(int(value)) for value in targets)
    return f"MOVEALL {values}"


def command_fraction(finger: int, permille: int) -> str:
    return f"FRAC {int(finger)} {int(permille)}"


def command_fraction_all(permilles: list[int | None]) -> str:
    values = " ".join("-1" if value is None else str(int(value)) for value in permilles)
    return f"FRACALL {values}"


def command_open(finger: int | None = None) -> str:
    return "OPEN" if finger is None else f"OPEN {int(finger)}"


def command_close(finger: int | None = None) -> str:
    return "CLOSE" if finger is None else f"CLOSE {int(finger)}"


def command_pulse(finger: int, direction: int, milliseconds: int) -> str:
    return f"PULSE {int(finger)} {int(direction)} {int(milliseconds)}"


def command_set_calibration(
    finger: int,
    soft_low: int,
    soft_high: int,
    increase_direction: int,
    extended_is_high: bool,
    max_move_ms: int,
) -> str:
    return (
        f"SET CAL {int(finger)} {int(soft_low)} {int(soft_high)} "
        f"{int(increase_direction)} {1 if extended_is_high else 0} {int(max_move_ms)}"
    )


def command_set_heartbeat(milliseconds: int) -> str:
    return f"SET HB {int(milliseconds)}"


def command_set_deadband(adc: int) -> str:
    return f"SET DEADBAND {int(adc)}"


def command_set_hold(enabled: bool) -> str:
    return f"SET HOLD {1 if enabled else 0}"


def command_set_legacy(enabled: bool) -> str:
    return f"SET LEGACY {1 if enabled else 0}"


def command_get_calibration() -> str:
    return "GET CAL"


def command_get_params() -> str:
    return "GET PARAM"


def finger_label(index: int) -> str:
    """로그·오류 메시지용 한국어 손가락 이름."""

    if 0 <= index < len(FINGER_LABELS_KO):
        return FINGER_LABELS_KO[index]
    return f"손가락{index}"
