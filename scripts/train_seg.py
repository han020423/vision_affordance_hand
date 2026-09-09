#!/usr/bin/env python
"""버전 관리되는 YAML 설정으로 재현 가능한 YOLO11n-seg 실험을 실행한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """파일 전체를 메모리에 올리지 않고 해시를 계산한다."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """실험 메타데이터를 읽고 저장소 상대경로인 학습 인자를 절대경로로 해석한다."""

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    experiment = dict(document.get("experiment", {}))
    training = dict(document.get("training", {}))
    required = {"model", "data", "epochs", "imgsz", "seed", "project", "name"}
    missing = required.difference(training)
    if missing:
        raise ValueError(f"학습 설정에 필수 항목이 없습니다: {sorted(missing)}")
    for key in ("model", "data", "project"):
        value = Path(str(training[key]))
        if not value.is_absolute():
            training[key] = str((REPOSITORY_ROOT / value).resolve())
    return experiment, training


def environment_metadata() -> dict[str, Any]:
    """모든 실행 시도에 사용한 버전과 인식 가능한 CUDA 장치를 기록한다."""

    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except OSError:
        driver = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names,
        "nvidia_driver_versions": driver,
        "ultralytics": importlib.metadata.version("ultralytics"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/public_pretrain.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="학습 데이터 5%로 1 epoch를 실행합니다. 실제 가중치 학습이 수행됩니다.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    experiment, training = load_config(args.config)
    data_path = Path(str(training["data"]))
    model_path = Path(str(training["model"]))
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터셋 YAML 파일이 없습니다: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"사전학습 가중치가 없습니다: {model_path}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA를 사용할 수 없어 의도하지 않은 CPU 학습을 중단합니다")

    # Ultralytics를 불러오기 전에 저장소 내부 설정 경로를 지정해야 사용자 홈에
    # settings.json을 쓰려는 동작과 권한 경고를 막을 수 있다.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPOSITORY_ROOT / ".ultralytics"))
    from ultralytics.cfg import get_cfg

    # 고정한 Ultralytics 설정 키가 바뀌었다면 긴 학습을 시작하기 전에 바로 실패시킨다.
    get_cfg(overrides=training)

    if args.smoke_test:
        training.update(
            {
                "epochs": 1,
                "fraction": 0.05,
                "name": str(training["name"]) + "_smoke",
                "patience": 0,
                "save_period": -1,
            }
        )
    metadata = {
        "experiment": experiment,
        "training": training,
        "environment": environment_metadata(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "dataset_yaml_sha256": sha256_file(data_path),
        "model_sha256": sha256_file(model_path),
        "mode": "dry_run" if args.dry_run else "smoke_test" if args.smoke_test else "train",
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("시험 실행: 설정만 해석했으며 학습은 시작하지 않았습니다.")
        return 0

    from ultralytics import YOLO

    project = Path(str(training["project"]))
    run_dir = project / str(training["name"])
    if run_dir.exists():
        raise FileExistsError(f"기존 학습 실행 디렉터리를 재사용하지 않습니다: {run_dir}")
    launch_dir = project / "_launches" / str(training["name"])
    launch_dir.mkdir(parents=True, exist_ok=False)
    launch_metadata_path = launch_dir / "launch_metadata.json"
    launch_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = launch_dir / "run_status.json"
    try:
        model = YOLO(str(model_path))
        model.train(**training)
    except BaseException as exc:
        status_path.write_text(
            json.dumps(
                {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    status_path.write_text('{"status": "completed"}\n', encoding="utf-8")
    actual_run_dir = Path(model.trainer.save_dir)
    shutil.copy2(launch_metadata_path, actual_run_dir / "launch_metadata.json")
    shutil.copy2(status_path, actual_run_dir / "run_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
