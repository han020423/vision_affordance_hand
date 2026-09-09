#!/usr/bin/env python
"""분리형 손 검출기(YOLO11n detect, 1클래스) 학습.

물체 모델(M)과 완전히 분리된 별도 모델이므로 물체 성능 오염 경로가 없다.
의수는 좌우가 고정된 물리 대상이므로 좌우 반전 증강을 끈다
(Copy-Paste 합성 데이터 생성 때와 같은 근거).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="hand_detector_v1_yolo11n_seed42")
    parser.add_argument("--project", type=Path, default=Path("outputs/training"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=args.seed, patience=15,
        project=str(args.project), name=args.name, exist_ok=False,
        fliplr=0.0, flipud=0.0,   # 의수 좌우/상하 고정
        verbose=True,
    )
    run_dir = Path(results.save_dir)
    status = {"status": "completed", "save_dir": str(run_dir),
              "data": str(args.data), "model": args.model,
              "epochs": args.epochs, "seed": args.seed}
    (run_dir / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
