#!/usr/bin/env python
"""공개 사전학습 또는 자체 미세조정용 서버 묶음을 결정적으로 생성한다."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PRETRAIN_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CLAUDE_CODE_HANDOFF.md"),
    Path("PROJECT_CONTEXT.md"),
    Path("requirements-data.txt"),
    Path("requirements-server.txt"),
    Path("configs"),
    Path("docs"),
    Path("scripts"),
    Path("src"),
    Path("tests"),
    Path("data/processed/yolo_public"),
    Path("models/pretrained/yolo11n-seg.pt"),
)
CUSTOM_FINETUNE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CLAUDE_CODE_HANDOFF.md"),
    Path("PROJECT_CONTEXT.md"),
    Path("requirements-data.txt"),
    Path("requirements-server.txt"),
    Path("configs"),
    Path("docs"),
    Path("scripts"),
    Path("src"),
    Path("tests"),
    Path("data/processed/yolo_custom_approved_v1"),
    Path("models/pretrained/yolo11n-seg.pt"),
    Path("outputs/training/public_pretrain_d_contain_background_seed42/weights/best.pt"),
)
EXCLUDED_SUFFIXES = {".pyc", ".cache", ".tmp"}
EXCLUDED_PARTS = {"__pycache__", ".ultralytics"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def selected_files(include_paths: tuple[Path, ...]) -> list[Path]:
    """선택한 학습 단계에 필요한 파일만 중복 없이 수집한다."""

    files: set[Path] = set()
    for relative in include_paths:
        source = REPOSITORY_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"서버 묶음에 필요한 입력 파일이 없습니다: {source}")
        candidates = [source] if source.is_file() else source.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative_path = path.relative_to(REPOSITORY_ROOT)
            if relative_path.suffix in EXCLUDED_SUFFIXES:
                continue
            if any(part in EXCLUDED_PARTS for part in relative_path.parts):
                continue
            files.add(relative_path)
    return sorted(files, key=lambda path: path.as_posix())


def normalized_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = mode
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("public-pretrain", "custom-finetune"),
        default="public-pretrain",
        help="서버에서 실행할 학습 단계를 선택합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="--output에 기존 생성 압축파일이 있으면 원자적으로 교체합니다.",
    )
    args = parser.parse_args()

    if args.profile == "custom-finetune":
        include_paths = CUSTOM_FINETUNE_PATHS
        bundle_root = "hjh_vision_hand"
        default_output = Path("artifacts/vision_hand_custom_finetune.tar.gz")
    else:
        include_paths = PUBLIC_PRETRAIN_PATHS
        bundle_root = "vision_hand_server_training"
        default_output = Path("artifacts/vision_hand_server_training.tar.gz")
    output_path = args.output or default_output

    files = selected_files(include_paths)
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "bundle_root": bundle_root,
        "files": [
            {
                "path": relative.as_posix(),
                "size": (REPOSITORY_ROOT / relative).stat().st_size,
                "sha256": sha256_file(REPOSITORY_ROOT / relative),
            }
            for relative in files
        ],
    }
    total_bytes = sum(record["size"] for record in manifest["files"])
    print(f"묶음 파일 수: {len(files)}")
    print(f"압축 전 바이트 수: {total_bytes}")
    if args.dry_run:
        print("시험 실행: 압축파일을 저장하지 않았습니다.")
        return 0
    if output_path.exists() and not args.replace:
        raise FileExistsError(f"기존 서버 묶음을 덮어쓰지 않습니다: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                root = bundle_root
                for relative in files:
                    source = REPOSITORY_ROOT / relative
                    name = f"{root}/{relative.as_posix()}"
                    mode = 0o755 if relative.suffix in {".py", ".sh"} else 0o644
                    info = normalized_info(name, source.stat().st_size, mode)
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
                info = normalized_info(f"{root}/bundle_manifest.json", len(manifest_bytes))
                archive.addfile(info, io.BytesIO(manifest_bytes))
    temporary.replace(output_path)
    digest = sha256_file(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    # Windows에서 묶음을 만들어도 Linux의 ``sha256sum -c``로 검사할 수 있도록
    # 체크섬 파일은 운영체제와 무관한 LF 줄바꿈으로 저장한다.
    checksum_path.write_bytes(f"{digest}  {output_path.name}\n".encode("ascii"))
    print(f"저장 완료: {output_path}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
