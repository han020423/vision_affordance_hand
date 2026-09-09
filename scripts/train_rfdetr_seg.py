#!/usr/bin/env python
"""버전 관리되는 YAML 설정으로 RF-DETR-Seg 미세조정(실험 K)을 실행한다.

train_seg.py(Ultralytics용)와 같은 재현성 규칙을 따른다.

- 설정 YAML과 데이터셋 manifest의 SHA-256을 launch_metadata.json에 기록한다.
- 기존 결과 폴더를 덮어쓰지 않는다.
- 성공·실패 여부를 run_status.json에 기록한다.
- CUDA가 없으면 --allow-cpu 없이는 실행을 중단한다.
- --dry-run은 rfdetr 패키지 없이도 설정과 데이터 구조만 검증한다.

주의: rfdetr의 train()은 Ultralytics와 달리 fraction 인자가 없어 smoke test는
전체 데이터 1 epoch로 수행한다(--smoke-test). smoke 수치를 성능으로 보고하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPLITS = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    """파일 전체를 메모리에 올리지 않고 SHA-256 해시를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """실험 메타데이터와 학습 설정을 읽고 상대 경로를 저장소 기준 절대 경로로 만든다."""

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    experiment = dict(document.get("experiment", {}))
    training = dict(document.get("training", {}))
    required = {"variant", "dataset_dir", "output_dir", "seed", "train_args"}
    missing = required.difference(training)
    if missing:
        raise ValueError(f"학습 설정에 필수 항목이 없습니다: {sorted(missing)}")
    for key in ("dataset_dir", "output_dir"):
        value = Path(str(training[key]))
        if not value.is_absolute():
            training[key] = str((REPOSITORY_ROOT / value).resolve())
    return experiment, training


def validate_dataset(dataset_dir: Path) -> dict[str, int]:
    """COCO 변환본의 세 분할과 어노테이션 파일 존재를 검증하고 이미지 수를 센다."""

    counts: dict[str, int] = {}
    for split in REQUIRED_SPLITS:
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"어노테이션 파일이 없습니다: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        counts[split] = len(document.get("images", []))
        if counts[split] == 0:
            raise ValueError(f"{split} 분할에 이미지가 없습니다: {annotation_path}")
    return counts


def environment_metadata() -> dict[str, Any]:
    """실행 환경의 버전과 CUDA 장치를 기록한다. torch가 없으면 그 사실을 기록한다."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        metadata["torch"] = "미설치"
    try:
        metadata["rfdetr"] = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        metadata["rfdetr"] = "미설치"
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/custom_finetune_k_rfdetr_seg.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true", help="설정과 데이터 구조만 검증")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 epoch만 실행해 파이프라인 정상 여부를 확인(수치는 성능으로 보고 금지)",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    experiment, training = load_config(config_path)
    dataset_dir = Path(str(training["dataset_dir"]))
    output_dir = Path(str(training["output_dir"]))
    if args.smoke_test:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")

    counts = validate_dataset(dataset_dir)
    manifest_path = dataset_dir / "rfdetr_export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"변환 manifest가 없습니다: {manifest_path}")

    plan = {
        "experiment_id": experiment.get("id", "알 수 없음"),
        "variant": training["variant"],
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "split_image_counts": counts,
        "train_args": dict(training["train_args"]),
        "seed": training["seed"],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run:
        print("dry-run 검증을 통과했습니다. 학습은 실행하지 않았습니다.")
        return 0

    if output_dir.exists():
        raise FileExistsError(f"기존 결과 폴더를 덮어쓰지 않습니다: {output_dir}")

    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA를 사용할 수 없어 의도하지 않은 CPU 학습을 중단합니다.")

    # 재현성: rfdetr가 seed 인자를 받지 않으므로 전역 seed를 직접 고정한다.
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass

    import rfdetr

    variant_name = str(training["variant"])
    if not hasattr(rfdetr, variant_name):
        raise ValueError(f"rfdetr에 없는 모델 변형입니다: {variant_name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    launch_metadata = {
        "experiment": experiment,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "split_image_counts": counts,
        "environment": environment_metadata(),
        "smoke_test": bool(args.smoke_test),
        "seed": seed,
    }
    with (output_dir / "launch_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(launch_metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    train_args = dict(training["train_args"])
    if args.smoke_test:
        train_args["epochs"] = 1
        train_args.pop("early_stopping", None)
    train_args["dataset_dir"] = str(dataset_dir)
    train_args["output_dir"] = str(output_dir)

    status_path = output_dir / "run_status.json"
    try:
        model = getattr(rfdetr, variant_name)()
        model.train(**train_args)
    except BaseException as error:
        with status_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"status": "failed", "error": repr(error)},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        raise
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump({"status": "completed"}, handle, ensure_ascii=False, indent=2)
    print("학습이 정상 종료되어 run_status.json에 completed를 기록했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
