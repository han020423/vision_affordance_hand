"""HandClient: 젯슨/PC에서 Brunel Hand를 다루는 상위 인터페이스.

연결 절차(잘못된 펌웨어·이전 상태를 가정하지 않는다):

    ID          펌웨어 이름·프로토콜 확인. 응답이 없거나 다르면 즉시 실패
    STOP        이전 상태 정리
    SET LEGACY 0  단일 문자 명령 차단(헤드리스 안전)
    SET HB <ms>   heartbeat 켜기(이동 중 호스트가 죽으면 펌웨어가 정지)
    SET CAL ×4    YAML 보정값 주입
    GET CAL       주입 결과 검증

사용 예:

    calibration = load_calibration(Path("configs/hardware/hand_actuators.yaml"))
    with HandClient(SerialLink("/dev/brunel_hand"), calibration) as hand:
        hand.open()
        results = hand.wait_settled()
        hand.move_fractions([0.5, 0.5, 0.5, None])   # 엄지 제외 반 접기

이동 중에는 wait_settled()가 주기적으로 PING을 보내 heartbeat를 유지한다.
어떤 경로로든 종료(with 블록 탈출, 예외)하면 STOP을 보낸다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import protocol
from .link import LinkError
from .presets import CalibrationError, HandCalibration, fraction_to_permille
from .protocol import Event, FingerStatus, Response


class HandError(RuntimeError):
    """손 제어 실패."""


@dataclass(frozen=True)
class MoveResult:
    """손가락 하나의 이동 종료 결과."""

    finger: int
    kind: str      # DONE | STALL | TIMEOUT | FAULT | (호스트 판단) HOST_TIMEOUT
    adc: int | None

    @property
    def reached(self) -> bool:
        return self.kind == "DONE"

    @property
    def blocked(self) -> bool:
        """무언가에 막혀 멈춘 경우.

        주의: 이것만으로 파지 성공을 판단할 수 없다. 2026-08-24 확인 결과
        얇은 물체는 손가락이 끝까지 닫히면서(`DONE`) 잡히고, 손가락이 구조상
        더 굽지 않는 자세(`WRAP`)에서도 `DONE`으로 끝난다. 반대로 `STALL`은
        머그 몸통처럼 두꺼운 물체에 걸렸을 때 나타난다.
        """

        return self.kind == "STALL"


class HandClient:
    """펌웨어와의 세션 하나를 관리한다."""

    def __init__(
        self,
        link,
        calibration: HandCalibration,
        *,
        response_timeout: float = 1.0,
        ping_interval_ms: int | None = None,
        event_log=None,
        clock=time.monotonic,
    ) -> None:
        self._link = link
        self._cal = calibration
        self._response_timeout = response_timeout
        # YAML의 heartbeat_ms는 호스트가 PING을 보내는 간격이다. 펌웨어 `SET HB`는
        # "이 시간 동안 아무 명령도 없으면 정지"하는 침묵 허용 시간이므로 더 크게 잡는다.
        self._ping_interval_ms = (
            calibration.heartbeat_ms if ping_interval_ms is None else ping_interval_ms
        )
        self._heartbeat_timeout_ms = max(3 * self._ping_interval_ms, 600)
        self._event_log = event_log            # callable(str) — 모든 송수신 기록
        self._clock = clock                    # 시험에서 가짜 시계를 주입한다
        self._pending_events: list[Event] = []
        self._connected = False
        self.identity: dict[str, str] | None = None   # connect() 성공 시 채워진다

    # ------------------------------------------------------------------ 기본 IO

    def _log(self, direction: str, text: str) -> None:
        if self._event_log is not None:
            self._event_log(f"{direction} {text}")

    def _send(self, command: str) -> None:
        self._log("TX", command)
        self._link.write_line(command)

    def _read(self, timeout: float) -> Response | Event | None:
        line = self._link.read_line(timeout)
        if line is None:
            return None
        self._log("RX", line)
        return protocol.parse_line(line)

    @staticmethod
    def _expected_kind(command: str) -> str:
        """명령에 대해 펌웨어가 돌려줄 `OK <종류>`의 종류 토큰."""

        tokens = command.split()
        if not tokens:
            return ""
        head = tokens[0].upper()
        if head == "PING":
            return "PONG"
        if head in {"SET", "GET"} and len(tokens) > 1:
            return tokens[1].upper()
        return head

    def _request(self, command: str, *, timeout: float | None = None) -> Response:
        """명령을 보내고 그 명령에 해당하는 Response가 올 때까지 기다린다.

        이벤트는 보류 목록에 쌓는다. 기대한 종류가 아닌 `OK` 줄은 이전 세션이나
        다른 프로세스가 남긴 것으로 보고 버린다. 이렇게 하지 않으면 엉뚱한 줄 하나에
        이후 모든 응답이 한 칸씩 밀린다(2026-08-24 젯슨에서 실제로 겪음).
        """

        self._send(command)
        expected = self._expected_kind(command)
        deadline = self._clock() + (self._response_timeout if timeout is None else timeout)
        while self._clock() < deadline:
            message = self._read(0.1)
            if message is None:
                continue
            if isinstance(message, Event):
                self._pending_events.append(message)
                continue
            if message.failed:
                raise HandError(f"명령 실패: {command} -> {message.raw}")
            if expected and message.kind != expected:
                self._log("무시", f"{message.raw} (기대: OK {expected})")
                continue
            return message
        raise HandError(f"응답 시간 초과: {command}")

    def poll_events(self, timeout: float = 0.0) -> list[Event]:
        """보류된 이벤트와 지금 도착한 이벤트를 모두 돌려준다(비차단에 가깝게)."""

        events, self._pending_events = self._pending_events, []
        message = self._read(timeout)
        while message is not None:
            if isinstance(message, Event):
                events.append(message)
            message = self._read(0.0)
        return events

    # ------------------------------------------------------------------ 연결

    def connect(self) -> dict[str, str]:
        """펌웨어 확인부터 보정 주입까지 수행한다. 반환값은 ID 정보."""

        # 이전 세션이 남긴 출력 정리. 전송 중인 줄이 더 있을 수 있으므로
        # 조용해질 때까지 반복해서 비운다.
        for _ in range(10):
            leftover = self._link.drain() if hasattr(self._link, "drain") else []
            extra = self._link.read_line(0.15)
            if extra is not None:
                leftover.append(extra)
            for line in leftover:
                self._log("RX(잔여)", line)
            if not leftover:
                break

        identity_response = self._request(protocol.command_identify())
        identity = protocol.parse_identity(identity_response)
        if identity.get("name") != protocol.FIRMWARE_NAME:
            raise HandError(f"다른 펌웨어입니다: {identity}")
        if int(identity.get("proto", -1)) != protocol.PROTOCOL_VERSION:
            raise HandError(f"프로토콜 버전 불일치: {identity}")

        self._request(protocol.command_stop())
        self._request(protocol.command_set_legacy(False))
        self._request(protocol.command_set_heartbeat(self._heartbeat_timeout_ms))
        self._request(protocol.command_set_deadband(self._cal.deadband_adc))
        self._request(protocol.command_set_hold(self._cal.hold_brake))
        for finger in self._cal.fingers:
            self._request(
                protocol.command_set_calibration(
                    finger.index,
                    finger.soft_low,
                    finger.soft_high,
                    finger.adc_increase_direction,
                    finger.extended_is_high,
                    finger.max_move_ms,
                )
            )
        echoed = protocol.parse_calibration_echo(self._request(protocol.command_get_calibration()))
        for finger in self._cal.fingers:
            expected = (
                finger.soft_low,
                finger.soft_high,
                finger.adc_increase_direction,
                finger.extended_is_high,
                finger.max_move_ms,
            )
            if echoed.get(finger.index) != expected:
                raise HandError(
                    f"보정 주입 검증 실패({finger.name}): 보냄 {expected}, 펌웨어 {echoed.get(finger.index)}"
                )
        self._connected = True
        self.identity = identity
        return identity

    def __enter__(self) -> "HandClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """항상 안전하게 끝낸다: STOP을 보내고 연결을 닫는다."""

        try:
            if self._connected:
                self._send(protocol.command_stop())
                # 응답은 최선 노력으로만 기다린다
                self._read(0.3)
        except (LinkError, HandError):
            pass
        finally:
            self._connected = False
            self._link.close()

    # ------------------------------------------------------------------ 조회

    def positions(self) -> list[int]:
        return protocol.parse_positions(self._request(protocol.command_positions()))

    def status(self) -> list[FingerStatus]:
        return protocol.parse_status(self._request(protocol.command_status()))

    def ping(self) -> None:
        """heartbeat 유지용. 이동 중 카메라 루프에서 주기적으로 호출한다."""

        self._request(protocol.command_ping(), timeout=0.5)

    def params(self) -> str:
        """펌웨어의 현재 설정값 원문(`OK PARAM ...`)."""

        return self._request(protocol.command_get_params()).raw

    def send_raw(self, command: str, *, collect_events: float = 0.0) -> tuple[Response, list[Event]]:
        """펌웨어 명령을 그대로 보낸다(진단용). 응답과 그 뒤 이벤트를 함께 돌려준다."""

        response = self._request(command)
        events = self.poll_events(timeout=collect_events) if collect_events > 0 else []
        return response, events

    # ------------------------------------------------------------------ 구동

    def stop(self) -> None:
        """즉시 전체 정지."""

        self._request(protocol.command_stop())

    def move_adc(self, finger: int, target: int) -> None:
        calibration = self._cal.finger(finger)
        if not calibration.contains(target):
            raise HandError(
                f"{calibration.name} 목표 {target}이 소프트 한계 "
                f"{calibration.soft_low}~{calibration.soft_high}를 벗어납니다."
            )
        self._request(protocol.command_move(finger, target))

    def move_fraction(self, finger: int, fraction: float) -> None:
        self._cal.finger(finger)  # 번호 검증
        self._request(protocol.command_fraction(finger, fraction_to_permille(fraction)))

    def move_fractions(self, fractions: list[float | None]) -> None:
        """네 손가락 동시 목표. None이면 그 손가락은 그대로 둔다."""

        if len(fractions) != len(self._cal.fingers):
            raise HandError(f"비율은 손가락 {len(self._cal.fingers)}개만큼 필요합니다.")
        permilles = [
            None if value is None else fraction_to_permille(value) for value in fractions
        ]
        self._request(protocol.command_fraction_all(permilles))

    def open(self, finger: int | None = None) -> None:
        self._request(protocol.command_open(finger))

    def close_hand(self, finger: int | None = None) -> None:
        self._request(protocol.command_close(finger))

    def move_preset(self, name: str) -> None:
        """프리셋의 모든 손가락을 동시에 이동한다.

        물체를 쥘 때는 `run_preset()`이나 `move_preset_step()`으로 순서를 지켜야 한다.
        동시에 닫으면 엄지가 받침이 되기 전에 손가락이 도착해 물체가 밀려난다(2026-08-24 실측).
        """

        fractions = self._cal.preset(name)
        self.move_fractions(list(fractions))

    def move_preset_step(self, name: str, step: int) -> tuple[int, ...]:
        """프리셋의 한 단계만 구동한다. 그 단계에 속한 손가락 번호를 돌려준다."""

        fractions = self._cal.preset(name)
        steps = self._cal.preset_steps(name)
        if not 0 <= step < len(steps):
            raise HandError(f"프리셋 {name}의 단계 번호가 범위를 벗어났습니다: {step}")
        group = steps[step]
        self.move_fractions(
            [fractions[i] if i in group else None for i in range(len(fractions))]
        )
        return group

    def run_preset(self, name: str, *, timeout: float = 8.0) -> dict[int, MoveResult]:
        """프리셋을 정의된 순서대로 끝까지 구동한다(블로킹).

        스크립트·CLI용이다. 카메라 루프에서는 GraspController가 단계별로 진행한다.
        """

        results: dict[int, MoveResult] = {}
        for step in range(len(self._cal.preset_steps(name))):
            group = self.move_preset_step(name, step)
            results.update(self.wait_settled(set(group), timeout=timeout))
        return results

    def move_pre_shape(self, pose: str) -> None:
        fractions = self._cal.pre_shape_preset(pose)
        self.move_fractions(list(fractions))

    # ------------------------------------------------------------------ 대기

    def wait_settled(
        self,
        fingers: set[int] | None = None,
        *,
        timeout: float = 6.0,
        ping_interval: float | None = None,
    ) -> dict[int, MoveResult]:
        """지정한 손가락들의 이동 종료 이벤트를 기다린다.

        기다리는 동안 ping_interval마다 PING을 보내 heartbeat를 유지한다.
        timeout까지 종료 이벤트가 없으면 STOP을 보내고 HOST_TIMEOUT으로 기록한다.
        """

        waiting = set(range(len(self._cal.fingers))) if fingers is None else set(fingers)
        results: dict[int, MoveResult] = {}
        interval = self._ping_interval_ms / 1000.0 if ping_interval is None else ping_interval
        deadline = self._clock() + timeout
        last_ping = self._clock()

        # STATUS로 이미 IDLE인 손가락(이동을 시작하지 않은 것)을 걸러낸다.
        for entry in self.status():
            if entry.index in waiting and entry.state not in {"MOVING", "WAIT"}:
                waiting.discard(entry.index)
                results[entry.index] = MoveResult(entry.index, entry.state, entry.adc)

        while waiting and self._clock() < deadline:
            for event in self.poll_events(timeout=0.05):
                if event.is_settle and event.finger in waiting:
                    waiting.discard(event.finger)
                    results[event.finger] = MoveResult(event.finger, event.kind, event.adc)
            now = self._clock()
            if now - last_ping >= interval:
                try:
                    self._request(protocol.command_ping(), timeout=0.5)
                except HandError:
                    pass  # PING 실패는 치명적이지 않다. 다음 순회에서 재시도.
                last_ping = now

        if waiting:
            self.stop()
            for finger in waiting:
                results[finger] = MoveResult(finger, "HOST_TIMEOUT", None)
        return results


def build_serial_client(
    port: str | None,
    calibration: HandCalibration,
    *,
    baud: int | None = None,
    event_log=None,
) -> HandClient:
    """포트 문자열로 SerialLink + HandClient를 만든다. port가 None이면 YAML 값 사용."""

    from .link import SerialLink

    resolved = port or calibration.port
    if not resolved:
        raise CalibrationError("시리얼 포트가 지정되지 않았습니다(--port 또는 YAML serial.port).")
    link = SerialLink(resolved, baud or calibration.baud)
    return HandClient(link, calibration, event_log=event_log)
