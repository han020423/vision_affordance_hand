#!/usr/bin/env python
"""파지 성공률 시험 기록 도구 (젯슨 실행용).

최종 평가의 비교 기준선과 제안 시스템 시도를 같은 형식의 CSV로 기록한다.

  --mode baseline1  : **비전 없음** 기준선. 물체와 무관하게 고정 프리셋(기본
                      POWER)으로만 파지한다. 손 하드웨어의 접촉(스톨) 적응은
                      동일하게 쓰므로, 이 기준선과 제안 시스템의 차이는
                      "비전이 고른 부위·자세"의 기여분이다.
  --mode log-only   : 제안 시스템 시도 기록용. 파지는 run_realtime_rfdetr_part.py
                      (+--hand-port)에서 수행하고, 이 도구는 성공/실패만 묻고
                      같은 CSV에 적는다.

성공 판정 기준(보고서와 동일): 파지 후 물체를 들어 올려 3초 유지하면 성공.
숫자는 실제 시도 결과만 기록한다 — 추정·보정 금지.

사용 예:
  .venv/bin/python scripts/run_baseline_grasp_trials.py --mode baseline1 \
      --port /dev/ttyACM0 --object mug_blue --trials 10
  .venv/bin/python scripts/run_baseline_grasp_trials.py --mode log-only \
      --object can_pepsi --trials 10 --system part_p
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

CSV_FIELDS = ["timestamp", "mode", "system", "object", "preset", "trial", "success", "note"]


def ask(prompt: str, choices: set[str]) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print(f"  입력은 {'/'.join(sorted(choices))} 중 하나여야 합니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline1", "log-only"], required=True)
    parser.add_argument("--object", required=True, help="물체 이름 (예: mug_blue, can_pepsi)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--preset", default="POWER", help="baseline1에서 쓸 고정 프리셋")
    parser.add_argument("--system", default="baseline1",
                        help="기록에 남길 시스템 이름 (log-only에서는 예: part_p)")
    parser.add_argument("--port", default=None, help="baseline1 시리얼 포트")
    parser.add_argument("--mock", action="store_true", help="모의 컨트롤러로 흐름 점검")
    parser.add_argument("--hand-config", default="configs/hardware/hand_actuators.yaml")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/grasp_trials/grasp_trials.csv"))
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output.exists()

    client = None
    if args.mode == "baseline1":
        if not (args.port or args.mock):
            print("baseline1은 --port 또는 --mock이 필요합니다.", file=sys.stderr)
            return 2
        from src.hand_control import HandClient, MockLink, SerialLink, load_calibration

        calibration = load_calibration(REPOSITORY_ROOT / args.hand_config)
        link = MockLink(calibration) if args.mock else SerialLink(args.port, calibration.baud)
        client = HandClient(link, calibration)
        identity = client.connect()
        print(f"손 펌웨어 확인: {identity.get('name')} fw {identity.get('fw')}")
        if args.preset not in calibration.available_presets():
            print(f"프리셋 {args.preset}이 보정 설정에 없습니다: "
                  f"{calibration.available_presets()}", file=sys.stderr)
            return 2

    print(f"[{args.mode}] 물체={args.object}, 목표 {args.trials}회, 기록={output}")
    print("성공 기준: 들어 올려 3초 유지. r=이 시도 무효(다시), q=중단 저장.")

    completed = 0
    rows = []
    try:
        while completed < args.trials:
            trial = completed + 1
            if args.mode == "baseline1":
                input(f"\n[{trial}/{args.trials}] 손을 펼칩니다. Enter → ")
                client.open()
                client.wait_settled()
                input("물체를 손 안 파지 위치에 놓고 Enter → ")
                print(f"고정 프리셋 {args.preset} 파지 실행...")
                client.run_preset(args.preset)
            else:
                input(f"\n[{trial}/{args.trials}] 실시간 창에서 파지를 수행한 뒤 Enter → ")
            answer = ask("들어 올려 3초 유지 성공? [y/n/r/q] → ", {"y", "n", "r", "q"})
            if answer == "q":
                break
            if answer == "r":
                print("  이 시도는 무효 처리(기록 안 함). 같은 번호로 다시.")
                if args.mode == "baseline1":
                    client.open()
                continue
            rows.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "mode": args.mode, "system": args.system,
                         "object": args.object,
                         "preset": args.preset if args.mode == "baseline1" else "",
                         "trial": trial, "success": 1 if answer == "y" else 0,
                         "note": ""})
            completed += 1
            if args.mode == "baseline1":
                input("물체를 회수하고 Enter (손 펼침) → ")
                client.open()
    finally:
        if client is not None:
            try:
                client.stop()
                client.open()
                client.shutdown()
            except Exception:
                pass
        with output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerows(rows)

    successes = sum(row["success"] for row in rows)
    print(f"\n기록 완료: {completed}회 중 성공 {successes}회 "
          f"({successes / completed * 100:.0f}%)" if completed else "\n기록 없음")
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
