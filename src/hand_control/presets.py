"""손 보정값·프리셋 설정(`configs/hardware/hand_actuators.yaml`) 로더.

보정값의 단일 진실은 YAML이고, 펌웨어에는 연결 시 `SET CAL`로 주입한다.
프리셋은 손가락별 **닫힘 비율 0.0(펼침)~1.0(접힘)** 으로 정의하므로 보정값이
바뀌어도 프리셋 정의는 그대로 쓸 수 있다.

실측하지 않은 값(`null`)이나 `calibration_confirmed: false`이면 구동을 거부한다.
사용자가 확인하지 않은 목표 위치로 모터를 돌리지 않기 위한 장치다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .protocol import FINGER_NAMES

FINGER_COUNT = 4


class CalibrationError(ValueError):
    """보정 설정이 없거나 구동에 쓸 수 없는 상태."""


@dataclass(frozen=True)
class FingerCalibration:
    """손가락 하나의 보정값."""

    index: int
    name: str
    soft_low: int
    soft_high: int
    adc_low_limit: int
    adc_high_limit: int
    adc_increase_direction: int      # 1 또는 2: 이 방향 번호로 구동하면 ADC 증가
    extended_is_high: bool           # 펼친 상태가 높은 ADC인지 (엄지만 False)
    max_move_ms: int

    @property
    def open_adc(self) -> int:
        """완전히 펼친 쪽의 소프트 한계."""

        return self.soft_high if self.extended_is_high else self.soft_low

    @property
    def closed_adc(self) -> int:
        """완전히 접힌 쪽의 소프트 한계."""

        return self.soft_low if self.extended_is_high else self.soft_high

    def fraction_to_adc(self, fraction: float) -> int:
        """닫힘 비율(0.0~1.0)을 목표 ADC로 바꾼다.

        펌웨어의 정수 연산(`open + span * permille / 1000`, 0으로 향한 절삭)과
        같은 값이 나오도록 계산한다. 실제 목표는 `FRAC` 명령으로 펌웨어가 정하며,
        이 함수는 로그와 검증에 쓴다.
        """

        permille = fraction_to_permille(fraction)
        span = self.closed_adc - self.open_adc
        return self.open_adc + int(span * permille / 1000)

    def adc_to_fraction(self, adc: int) -> float:
        """현재 ADC를 닫힘 비율로 바꾼다(소프트 한계 밖이면 0~1을 넘을 수 있다)."""

        span = self.closed_adc - self.open_adc
        if span == 0:
            return 0.0
        return (adc - self.open_adc) / span

    def clamp(self, adc: int) -> int:
        """소프트 한계 안으로 자른다."""

        return max(self.soft_low, min(self.soft_high, int(adc)))

    def contains(self, adc: int) -> bool:
        return self.soft_low <= adc <= self.soft_high


@dataclass(frozen=True)
class HandCalibration:
    """손 전체 설정."""

    fingers: tuple[FingerCalibration, ...]
    port: str | None
    baud: int
    heartbeat_ms: int
    deadband_adc: int
    hold_brake: bool
    presets: dict[str, tuple[float, ...] | None]
    pre_shape: dict[str, tuple[float, ...] | None]
    preset_sequence: dict[str, tuple[tuple[int, ...], ...]]
    source_path: Path

    def finger(self, index: int) -> FingerCalibration:
        if not 0 <= index < len(self.fingers):
            raise CalibrationError(f"손가락 번호는 0~{len(self.fingers) - 1}입니다: {index}")
        return self.fingers[index]

    def available_presets(self) -> list[str]:
        """비율이 실제로 채워진 프리셋 이름만 돌려준다."""

        return sorted(name for name, values in self.presets.items() if values is not None)

    def preset(self, name: str) -> tuple[float, ...]:
        """프리셋 비율을 돌려준다. 미정이면 CalibrationError."""

        key = name.upper()
        if key not in self.presets:
            raise CalibrationError(
                f"모르는 프리셋입니다: {name} (사용 가능: {', '.join(self.available_presets()) or '없음'})"
            )
        values = self.presets[key]
        if values is None:
            raise CalibrationError(
                f"프리셋 '{key}'의 손가락별 닫힘 비율이 아직 정해지지 않았습니다. "
                f"{self.source_path.name}의 presets를 실측 후 채워야 합니다."
            )
        return values

    def preset_steps(self, name: str) -> tuple[tuple[int, ...], ...]:
        """프리셋을 구동할 손가락 그룹 순서. 정의가 없으면 네 손가락 동시.

        물체를 쥘 때는 엄지를 먼저 세워야 하므로(2026-08-24 실측) 순서가 중요하다.
        """

        self.preset(name)   # 비율이 정의됐는지 먼저 검사
        steps = self.preset_sequence.get(name.upper())
        if not steps:
            return (tuple(range(len(self.fingers))),)
        return steps

    def pre_shape_preset(self, pose: str) -> tuple[float, ...]:
        """파지 직전 반개방 자세. 미정이면 CalibrationError."""

        key = pose.upper()
        if key not in self.pre_shape:
            raise CalibrationError(f"모르는 자세입니다: {pose}")
        values = self.pre_shape[key]
        if values is None:
            raise CalibrationError(
                f"PRE_SHAPE '{key}'가 아직 정해지지 않았습니다. {self.source_path.name}을 채워야 합니다."
            )
        return values


def fraction_to_permille(fraction: float) -> int:
    """0.0~1.0 비율을 펌웨어가 받는 천분율 정수로 바꾼다."""

    if not 0.0 <= fraction <= 1.0:
        raise CalibrationError(f"닫힘 비율은 0.0~1.0이어야 합니다: {fraction}")
    return int(round(fraction * 1000))


def _require(value, field: str, where: str):
    if value is None:
        raise CalibrationError(
            f"{where}의 '{field}'가 아직 실측되지 않았습니다(null). 실측값을 채운 뒤 사용하세요."
        )
    return value


def _parse_fraction_list(values, where: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != FINGER_COUNT:
        raise CalibrationError(f"{where}는 손가락 {FINGER_COUNT}개의 비율 목록이어야 합니다: {values!r}")
    parsed = []
    for value in values:
        if value is None:
            raise CalibrationError(f"{where}에 null이 있습니다. 실측 후 채우세요.")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise CalibrationError(f"{where}의 비율은 0.0~1.0이어야 합니다: {number}")
        parsed.append(number)
    return tuple(parsed)


def load_calibration(path: Path, *, require_confirmed: bool = True) -> HandCalibration:
    """YAML을 읽어 HandCalibration을 만든다.

    require_confirmed가 True면 `calibration_confirmed: true`가 아닐 때 거부한다.
    """

    path = Path(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if require_confirmed and not document.get("calibration_confirmed", False):
        raise CalibrationError(
            f"{path.name}의 calibration_confirmed가 true가 아닙니다. "
            "사용자가 방향과 한계를 확인한 뒤에만 구동할 수 있습니다."
        )

    serial_section = dict(document.get("serial", {}) or {})
    control = dict(document.get("control", {}) or {})

    entries = document.get("fingers") or []
    if len(entries) != FINGER_COUNT:
        raise CalibrationError(f"fingers 항목은 {FINGER_COUNT}개여야 합니다: {len(entries)}개")

    fingers: list[FingerCalibration] = []
    for position, entry in enumerate(entries):
        entry = dict(entry or {})
        index = int(entry.get("index", position))
        if index != position:
            raise CalibrationError(f"fingers[{position}]의 index가 {index}입니다. 순서를 지켜야 합니다.")
        name = str(entry.get("name") or FINGER_NAMES[position])
        where = f"fingers[{position}]({name})"
        soft_low = int(_require(entry.get("soft_low"), "soft_low", where))
        soft_high = int(_require(entry.get("soft_high"), "soft_high", where))
        if soft_low >= soft_high:
            raise CalibrationError(f"{where}: soft_low({soft_low}) < soft_high({soft_high})여야 합니다.")
        increase_direction = int(_require(entry.get("adc_increase_direction"), "adc_increase_direction", where))
        if increase_direction not in (1, 2):
            raise CalibrationError(f"{where}: adc_increase_direction은 1 또는 2입니다: {increase_direction}")
        extended_is_high = bool(_require(entry.get("extended_is_high"), "extended_is_high", where))
        max_move_ms = int(_require(entry.get("max_move_ms"), "max_move_ms", where))
        low_limit = int(entry.get("adc_low_limit", soft_low))
        high_limit = int(entry.get("adc_high_limit", soft_high))
        fingers.append(
            FingerCalibration(
                index=index,
                name=name,
                soft_low=soft_low,
                soft_high=soft_high,
                adc_low_limit=low_limit,
                adc_high_limit=high_limit,
                adc_increase_direction=increase_direction,
                extended_is_high=extended_is_high,
                max_move_ms=max_move_ms,
            )
        )

    raw_sequence = dict(document.get("preset_sequence", {}) or {})
    preset_sequence: dict[str, tuple[tuple[int, ...], ...]] = {}
    for name, steps in raw_sequence.items():
        key = str(name).upper()
        if not isinstance(steps, (list, tuple)) or not steps:
            raise CalibrationError(f"preset_sequence.{name}은 손가락 그룹 목록이어야 합니다: {steps!r}")
        parsed_steps = []
        seen: set[int] = set()
        for group in steps:
            if not isinstance(group, (list, tuple)) or not group:
                raise CalibrationError(f"preset_sequence.{name}의 그룹이 비었습니다: {group!r}")
            indices = []
            for value in group:
                index = int(value)
                if not 0 <= index < FINGER_COUNT:
                    raise CalibrationError(f"preset_sequence.{name}: 손가락 번호 0~{FINGER_COUNT-1}이어야 합니다: {index}")
                if index in seen:
                    raise CalibrationError(f"preset_sequence.{name}: 손가락 {index}가 두 번 나옵니다.")
                seen.add(index)
                indices.append(index)
            parsed_steps.append(tuple(indices))
        if len(seen) != FINGER_COUNT:
            missing = sorted(set(range(FINGER_COUNT)) - seen)
            raise CalibrationError(f"preset_sequence.{name}: 손가락 {missing}가 빠졌습니다.")
        preset_sequence[key] = tuple(parsed_steps)

    raw_presets = dict(document.get("presets", {}) or {})
    pre_shape_raw = dict(raw_presets.pop("PRE_SHAPE", {}) or {})
    presets: dict[str, tuple[float, ...] | None] = {}
    for name, values in raw_presets.items():
        presets[str(name).upper()] = _parse_fraction_list(values, f"presets.{name}")
    pre_shape: dict[str, tuple[float, ...] | None] = {}
    for pose, values in pre_shape_raw.items():
        pre_shape[str(pose).upper()] = _parse_fraction_list(values, f"presets.PRE_SHAPE.{pose}")

    return HandCalibration(
        fingers=tuple(fingers),
        port=serial_section.get("port"),
        baud=int(serial_section.get("baud", 115200)),
        heartbeat_ms=int(serial_section.get("heartbeat_ms", 300)),
        deadband_adc=int(control.get("deadband_adc", 20)),
        hold_brake=bool(control.get("hold_brake", True)),
        presets=presets,
        pre_shape=pre_shape,
        preset_sequence=preset_sequence,
        source_path=path,
    )
