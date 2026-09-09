#!/usr/bin/env python
"""학습을 시작하지 않고 서버와 데이터셋 준비 상태를 빠르게 점검한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS = "8.3.163"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/public_pretrain.yaml"),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    version = importlib.metadata.version("ultralytics")
    if version != EXPECTED_ULTRALYTICS:
        errors.append(f"ultralytics 버전이 {version}이며 예상 버전은 {EXPECTED_ULTRALYTICS}입니다")
    if not torch.cuda.is_available() and not args.allow_cpu:
        errors.append("PyTorch에서 CUDA를 사용할 수 없습니다")
    if torch.version.cuda is None:
        warnings.append("설치된 PyTorch가 CPU 전용 빌드입니다")

    with args.config.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    training = document["training"]
    paths = {
        "dataset_yaml": REPOSITORY_ROOT / str(training["data"]),
        "model": REPOSITORY_ROOT / str(training["model"]),
        "dataset_version": REPOSITORY_ROOT / str(document["experiment"]["dataset_version"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"필수 파일이 없습니다({name}): {path}")

    dataset_root = paths["dataset_yaml"].parent
    validator = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts/validate_yolo_dataset.py"), str(dataset_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if validator.returncode != 0:
        errors.append("YOLO 데이터셋 검증에 실패했습니다")

    gpu_info: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpu_info.append(
            {
                "index": index,
                "name": props.name,
                "vram_gib": round(props.total_memory / 1024**3, 2),
            }
        )
    free_bytes = shutil.disk_usage(REPOSITORY_ROOT).free
    report = {
        "status": "failed" if errors else "passed",
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu_info,
        "ultralytics": version,
        "free_disk_gib": round(free_bytes / 1024**3, 2),
        "dataset_validator_stdout": validator.stdout.strip(),
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
