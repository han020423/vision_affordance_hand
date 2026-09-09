"""손 컨트롤러와의 줄 단위 통신 계층.

- `SerialLink`: 실제 USB 시리얼(pyserial). 수신은 배경 스레드가 큐에 넣는다.
- `MockLink`: 펌웨어 동작을 흉내내어 하드웨어 없이 상위 코드를 시험한다.

두 구현은 같은 인터페이스를 갖는다.

    write_line(text)              보낼 줄 하나(줄바꿈은 구현이 붙인다)
    read_line(timeout) -> str|None  받은 줄 하나. 시간 안에 없으면 None
    close()

`read_line`은 timeout 안에서 기다렸다가 없으면 None을 돌려준다. 상위 코드가
따로 sleep하지 않아도 되므로 실제 시리얼과 모의 구현의 대기 방식 차이를
상위에서 신경 쓰지 않는다.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable, Sequence

from .presets import HandCalibration


class LinkError(RuntimeError):
    """통신 계층 오류."""


class SerialLink:
    """pyserial 기반 실제 연결.

    포트는 `exclusive=True`로 열어 두 프로세스가 같은 포트를 잡는 것을 막는다.
    1200 baud는 SAMD 보드에서 부트로더 진입 신호이므로 허용하지 않는다.
    """

    def __init__(self, port: str, baud: int = 115200, *, read_timeout: float = 0.1) -> None:
        if int(baud) == 1200:
            raise LinkError("1200 baud는 SAMD 부트로더 진입 신호이므로 사용할 수 없습니다.")
        try:
            import serial  # 지연 임포트: MockLink만 쓸 때는 pyserial이 없어도 된다
        except ImportError as error:  # pragma: no cover
            raise LinkError("pyserial이 필요합니다: pip install -r requirements-runtime.txt") from error

        self._port_name = port
        try:
            self._serial = serial.Serial(port, int(baud), timeout=read_timeout, exclusive=True)
        except serial.SerialException as error:
            raise LinkError(
                f"포트를 열 수 없습니다: {port} ({error}). "
                "Arduino IDE 시리얼 모니터나 다른 프로그램이 잡고 있는지 확인하세요."
            ) from error

        self._lines: queue.Queue[str] = queue.Queue()
        self._closing = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, name="hand-serial-reader", daemon=True)
        self._reader.start()

    @property
    def description(self) -> str:
        return f"{self._port_name}@{self._serial.baudrate}"

    def _read_loop(self) -> None:
        buffer = bytearray()
        while not self._closing.is_set():
            try:
                chunk = self._serial.read(64)
            except Exception:  # 포트 소실 등: 루프를 끝내고 상위 타임아웃에 맡긴다
                break
            if not chunk:
                continue
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = bytes(buffer).partition(b"\n")
                buffer = bytearray(rest)
                text = raw.decode("utf-8", "replace").strip()
                if text:
                    self._lines.put(text)

    def write_line(self, text: str) -> None:
        try:
            self._serial.write((text + "\n").encode("utf-8"))
            self._serial.flush()
        except Exception as error:
            raise LinkError(f"전송 실패: {error}") from error

    def read_line(self, timeout: float = 0.1) -> str | None:
        try:
            return self._lines.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def drain(self) -> list[str]:
        """큐에 남은 줄을 모두 비우고 돌려준다."""

        drained = []
        while True:
            try:
                drained.append(self._lines.get_nowait())
            except queue.Empty:
                return drained

    def close(self) -> None:
        self._closing.set()
        try:
            self._reader.join(timeout=1.0)
        finally:
            try:
                self._serial.close()
            except Exception:
                pass


class MockLink:
    """펌웨어를 흉내내는 모의 연결.

    하드웨어 없이 상위 코드(HandClient, 상태 머신, CLI)를 시험하기 위한 것이다.
    실제 측정한 이동 속도(접힘 방향이 빠르다)를 반영하고, `blocked`에 넣은
    손가락은 움직이지 않아 `EVT STALL`이 뜨게 한다.

    시간은 `clock`/`advance`로 주입한다. 시험에서는 가짜 시계를 넣어
    실제로 기다리지 않고 결정적으로 검증한다.
    """

    def __init__(
        self,
        calibration: HandCalibration,
        *,
        clock: Callable[[], float] = time.monotonic,
        advance: Callable[[float], None] = time.sleep,
        initial: Sequence[int] | None = None,
        blocked: Iterable[int] = (),
        counts_per_second_toward_low: float = 4200.0,
        counts_per_second_toward_high: float = 2400.0,
        legacy_enabled: bool = True,
    ) -> None:
        self._cal = calibration
        self._clock = clock
        self._advance = advance
        self._speed_low = counts_per_second_toward_low
        self._speed_high = counts_per_second_toward_high
        self.blocked = set(blocked)

        count = len(calibration.fingers)
        if initial is None:
            positions = [finger.open_adc for finger in calibration.fingers]
        else:
            positions = [int(value) for value in initial]
            if len(positions) != count:
                raise LinkError(f"initial은 손가락 {count}개여야 합니다.")
        self._adc = [float(value) for value in positions]
        self._target: list[int | None] = [None] * count
        self._move_start: list[float] = [0.0] * count
        self._progress_time: list[float] = [0.0] * count
        self._progress_adc: list[float] = [0.0] * count
        self._state = ["IDLE"] * count

        self.deadband = calibration.deadband_adc
        self.heartbeat_ms = 0
        self.hold = False
        self.legacy_enabled = legacy_enabled
        self.stall_window_ms = 200
        self.stall_min_delta = 60

        self._out: queue.Queue[str] = queue.Queue()
        self._last_rx = clock()
        self._boot_time = clock()
        self._emit(f"EVT BOOT hand_closed_loop fw 2")

    # ------------------------------------------------------------- 내부 도우미

    @property
    def description(self) -> str:
        return "mock"

    def _emit(self, text: str) -> None:
        self._out.put(text)

    def positions(self) -> list[int]:
        """시험 코드에서 현재 모의 위치를 확인할 때 쓴다."""

        self._pump()
        return [int(round(value)) for value in self._adc]

    def _pump(self) -> None:
        """현재 시각까지 모의 이동을 진행한다."""

        now = self._clock()
        for index, target in enumerate(self._target):
            if target is None:
                continue
            finger = self._cal.fingers[index]
            elapsed_since_start = (now - self._move_start[index]) * 1000.0
            adc = self._adc[index]
            error = target - adc

            if abs(error) <= self.deadband:
                self._finish(index, "DONE", now)
                continue

            if index in self.blocked:
                if (now - self._progress_time[index]) * 1000.0 >= self.stall_window_ms:
                    self._finish(index, "STALL", now)
                continue

            direction = 1.0 if error > 0 else -1.0
            speed = self._speed_high if direction > 0 else self._speed_low
            step = direction * speed * max(0.0, now - self._progress_time[index])
            if abs(step) > abs(error):
                step = error
            moved = self._adc[index] + step
            self._adc[index] = max(
                finger.adc_low_limit, min(finger.adc_high_limit, moved)
            )
            if abs(self._adc[index] - self._progress_adc[index]) >= self.stall_min_delta:
                self._progress_adc[index] = self._adc[index]
                self._progress_time[index] = now
            elif (now - self._progress_time[index]) * 1000.0 >= self.stall_window_ms:
                self._finish(index, "STALL", now)
                continue

            if elapsed_since_start >= finger.max_move_ms:
                self._finish(index, "TIMEOUT", now)

    def _finish(self, index: int, kind: str, now: float) -> None:
        self._target[index] = None
        self._state[index] = kind
        self._emit(f"EVT {kind} {index} {int(round(self._adc[index]))}")

    def _start_move(self, index: int, target: int) -> bool:
        finger = self._cal.fingers[index]
        if not finger.contains(target):
            self._emit(f"ERR RANGE {finger.name}")
            return False
        now = self._clock()
        self._target[index] = target
        self._state[index] = "MOVING"
        self._move_start[index] = now
        self._progress_time[index] = now
        self._progress_adc[index] = self._adc[index]
        return True

    # ------------------------------------------------------------- Link 인터페이스

    def write_line(self, text: str) -> None:
        self._pump()
        self._last_rx = self._clock()
        line = text.strip()
        if not line:
            return
        tokens = line.split()
        head = tokens[0].upper()

        if head == "ID":
            self._emit(
                "OK ID hand_closed_loop fw 2 proto 1 fingers "
                f"{len(self._cal.fingers)} adcbits 12 uptime "
                f"{int((self._clock() - self._boot_time) * 1000)}"
            )
        elif head == "PING":
            self._emit(f"OK PONG {int((self._clock() - self._boot_time) * 1000)}")
        elif head == "STOP" or line in {"s", "S"}:
            for index in range(len(self._cal.fingers)):
                self._target[index] = None
                self._state[index] = "IDLE"
            self._emit("OK STOP")
        elif head == "POS":
            self._emit("OK POS " + " ".join(str(int(round(value))) for value in self._adc))
        elif head == "STATUS":
            parts = []
            for index, value in enumerate(self._adc):
                target = self._target[index]
                parts.append(
                    f"{index}:{self._state[index]}:{int(round(value))}:"
                    f"{'-' if target is None else target}"
                )
            self._emit("OK STATUS " + " ".join(parts))
        elif head == "MOVE":
            index, target = int(tokens[1]), int(tokens[2])
            if self._start_move(index, target):
                self._emit(f"OK MOVE {index} {target}")
        elif head == "FRAC":
            index, permille = int(tokens[1]), int(tokens[2])
            target = self._cal.fingers[index].fraction_to_adc(permille / 1000.0)
            if self._start_move(index, target):
                self._emit(f"OK FRAC {index} {target}")
        elif head in {"MOVEALL", "FRACALL"}:
            values = [int(token) for token in tokens[1:]]
            targets: list[int | None] = []
            for index, value in enumerate(values):
                if value < 0:
                    targets.append(None)
                    continue
                if head == "FRACALL":
                    targets.append(self._cal.fingers[index].fraction_to_adc(value / 1000.0))
                else:
                    targets.append(value)
            for index, target in enumerate(targets):
                if target is not None and not self._cal.fingers[index].contains(target):
                    self._emit(f"ERR RANGE {self._cal.fingers[index].name}")
                    return
            for index, target in enumerate(targets):
                if target is not None:
                    self._start_move(index, target)
            self._emit(
                f"OK {head} " + " ".join("-1" if t is None else str(t) for t in targets)
            )
        elif head in {"OPEN", "CLOSE"}:
            fraction = 0.0 if head == "OPEN" else 1.0
            if len(tokens) > 1:
                index = int(tokens[1])
                target = self._cal.fingers[index].fraction_to_adc(fraction)
                if self._start_move(index, target):
                    self._emit(f"OK {head} {index} {target}")
            else:
                targets = [finger.fraction_to_adc(fraction) for finger in self._cal.fingers]
                for index, target in enumerate(targets):
                    self._start_move(index, target)
                self._emit(f"OK {head} " + " ".join(str(t) for t in targets))
        elif head == "SET":
            self._handle_set(tokens)
        elif head == "GET":
            self._handle_get(tokens)
        elif head == "HELP":
            self._emit("OK HELP")
        else:
            self._emit(f"ERR UNKNOWN {tokens[0]}")

    def _handle_set(self, tokens: list[str]) -> None:
        what = tokens[1].upper() if len(tokens) > 1 else ""
        if what == "CAL":
            self._emit(f"OK CAL {int(tokens[2])}")
        elif what == "HB":
            self.heartbeat_ms = int(tokens[2])
            self._emit(f"OK HB {self.heartbeat_ms}")
        elif what == "DEADBAND":
            self.deadband = int(tokens[2])
            self._emit(f"OK DEADBAND {self.deadband}")
        elif what == "HOLD":
            self.hold = tokens[2] != "0"
            self._emit(f"OK HOLD {1 if self.hold else 0}")
        elif what == "LEGACY":
            self.legacy_enabled = tokens[2] != "0"
            self._emit(f"OK LEGACY {1 if self.legacy_enabled else 0}")
        else:
            self._emit("ERR ARG SET CAL|HB|DEADBAND|HOLD|LEGACY")

    def _handle_get(self, tokens: list[str]) -> None:
        what = tokens[1].upper() if len(tokens) > 1 else ""
        if what == "CAL":
            parts = [
                f"{f.index}:{f.soft_low}:{f.soft_high}:{f.adc_increase_direction}:"
                f"{1 if f.extended_is_high else 0}:{f.max_move_ms}"
                for f in self._cal.fingers
            ]
            self._emit("OK CAL " + " ".join(parts))
        elif what == "PARAM":
            self._emit(
                f"OK PARAM deadband={self.deadband} hb={self.heartbeat_ms} "
                f"hold={1 if self.hold else 0} legacy={1 if self.legacy_enabled else 0} "
                f"stall_window={self.stall_window_ms} stall_delta={self.stall_min_delta} "
                "rest=500 loop=5"
            )
        else:
            self._emit("ERR ARG GET CAL|PARAM")

    def read_line(self, timeout: float = 0.1) -> str | None:
        self._pump()
        if self._out.empty() and timeout > 0:
            # 대기 시간만큼 시간을 흘려보내고 다시 진행한다(가짜 시계에서도 동작).
            self._advance(timeout)
            self._check_heartbeat()
            self._pump()
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def _check_heartbeat(self) -> None:
        if self.heartbeat_ms <= 0:
            return
        if not any(target is not None for target in self._target):
            return
        if (self._clock() - self._last_rx) * 1000.0 > self.heartbeat_ms:
            for index in range(len(self._cal.fingers)):
                self._target[index] = None
                self._state[index] = "IDLE"
            self._emit("EVT HEARTBEAT_STOP")
            self._last_rx = self._clock()

    def drain(self) -> list[str]:
        self._pump()
        drained = []
        while True:
            try:
                drained.append(self._out.get_nowait())
            except queue.Empty:
                return drained

    def close(self) -> None:
        return None
