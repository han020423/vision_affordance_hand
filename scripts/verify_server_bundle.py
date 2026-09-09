#!/usr/bin/env python
"""압축 해제한 서버 묶음의 모든 파일을 내장 SHA-256 manifest와 대조한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    manifest_path = ROOT / "bundle_manifest.json"
    if not manifest_path.is_file():
        print(f"오류: manifest 파일 누락 {manifest_path}", file=sys.stderr)
        return 1
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in document["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"파일 누락: {record['path']}")
            continue
        if path.stat().st_size != record["size"]:
            errors.append(f"파일 크기 불일치: {record['path']}")
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            errors.append(f"SHA-256 불일치: {record['path']}")
    if errors:
        for error in errors:
            print(f"오류: {error}", file=sys.stderr)
        return 1
    print(f"서버 묶음 파일 {len(document['files'])}개를 검증했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
