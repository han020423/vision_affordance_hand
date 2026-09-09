"""2클래스 YOLO 변환본을 파지 종류 3클래스로 재매핑한다.

기존 `export_public_yolo`가 만든 변환본의 폴리곤 기하는 그대로 두고,
각 폴리곤의 클래스만 원본 라벨 정보로 재분류한다.

새 클래스:
  0: handle_grasp_region  (손잡이형 파지 → PRECISION/WRAP 후보)
  1: body_grasp_region    (몸통형 파지 → POWER)
  2: functional_region    (기능 부위, 회피 대상)

재분류 근거:
  - UMD: 원본 .mat의 gt_label에서 grasp(1) 대 wrap-grasp(7) 다수결
  - 자체 데이터: 승인 manifest의 component `part` 필드 (handle/body)
  - Aff-Grasp: graspable에 세부 구분이 없어 handle_grasp로 간주하고
    가정 사실을 감사 기록에 남긴다 (사람 표본 점검 대상)
  - 기존 functional_region 폴리곤은 전부 클래스 2로 이동

원본 데이터와 기존 2클래스 변환본은 수정하지 않는다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy.io import loadmat

GRASP_TYPE_NAMES: dict[int, str] = {
    0: "handle_grasp_region",
    1: "body_grasp_region",
    2: "functional_region",
}

# UMD 원본 값: 1=grasp(손잡이형), 7=wrap-grasp(몸통형)
UMD_HANDLE_SOURCE_VALUE = 1
UMD_BODY_SOURCE_VALUE = 7
# IIT-AFF 원본 값: 5=grasp(손잡이형), 9=w-grasp(몸통형)
IIT_HANDLE_SOURCE_VALUE = 5
IIT_BODY_SOURCE_VALUE = 9

# 다수결에서 소수 비율이 이 값을 넘으면 모호 표시를 남긴다(배정은 다수결 유지).
AMBIGUOUS_MINORITY_RATIO = 0.4


def remap_label_line(line: str, new_class_id: int) -> str:
    """YOLO 라벨 한 줄의 클래스 토큰만 새 값으로 바꾼다."""

    fields = line.split()
    if len(fields) < 7:
        raise ValueError(f"폴리곤 좌표가 부족한 라벨 줄입니다: {line[:60]}")
    return " ".join([str(int(new_class_id)), *fields[1:]])


def component_grasp_type_vote(
    raw_mask: np.ndarray,
    instance_mask: np.ndarray,
    component_id: int,
    handle_value: int,
    body_value: int,
) -> tuple[int | None, dict[str, Any]]:
    """컴포넌트의 파지 종류를 원본 픽셀 다수결로 정한다."""

    if raw_mask.shape != instance_mask.shape:
        return None, {"reason": "raw_instance_size_mismatch"}
    selected = raw_mask[instance_mask == component_id]
    handle_votes = int(np.count_nonzero(selected == handle_value))
    body_votes = int(np.count_nonzero(selected == body_value))
    total = handle_votes + body_votes
    audit: dict[str, Any] = {
        "handle_votes": handle_votes,
        "body_votes": body_votes,
        "vote_total": total,
    }
    if total == 0:
        audit["reason"] = "no_grasp_source_pixels"
        return None, audit
    new_class = 0 if handle_votes >= body_votes else 1
    minority_ratio = min(handle_votes, body_votes) / total
    audit["minority_ratio"] = round(minority_ratio, 4)
    audit["ambiguous"] = minority_ratio > AMBIGUOUS_MINORITY_RATIO
    return new_class, audit


def umd_component_grasp_type(
    raw_mask: np.ndarray, instance_mask: np.ndarray, component_id: int
) -> tuple[int | None, dict[str, Any]]:
    """UMD 컴포넌트의 파지 종류를 gt_label 다수결로 정한다."""

    return component_grasp_type_vote(
        raw_mask, instance_mask, component_id,
        UMD_HANDLE_SOURCE_VALUE, UMD_BODY_SOURCE_VALUE,
    )


def custom_component_grasp_type(component: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """자체 데이터 컴포넌트의 파지 종류를 검수된 part 필드로 정한다."""

    part = str(component.get("part", ""))
    if part == "handle":
        return 0, {"part": part}
    if part == "body":
        return 1, {"part": part}
    return None, {"part": part, "reason": "unknown_custom_part"}


def _load_source_index(manifest_paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """(source_dataset, source_id)로 원본 manifest 레코드를 찾는 색인을 만든다."""

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in manifest_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (str(record.get("source_dataset", "")), str(record.get("source_id", "")))
                index[key] = record
    return index


def _resolve(repository_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path


def remap_export_to_grasp_type(
    export_root: Path,
    source_manifest_paths: list[Path],
    repository_root: Path,
) -> dict[str, Any]:
    """변환본의 라벨 클래스를 3클래스로 재매핑하고 감사 기록을 남긴다.

    export_root는 `export_public_yolo`가 방금 생성한 폴더여야 하며, 이 함수는
    그 안의 라벨 txt와 dataset.yaml, dataset_version.json만 수정한다.
    """

    manifest_path = export_root / "export_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"export_manifest.jsonl이 없습니다: {manifest_path}")
    source_index = _load_source_index(source_manifest_paths)

    counters: Counter[str] = Counter()
    class_instances: Counter[str] = Counter()
    audit_records: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        export_records = [json.loads(line) for line in handle if line.strip()]

    for record in export_records:
        if not record.get("exports"):
            continue
        source_dataset = str(record.get("source_dataset", ""))
        source_id = str(record.get("source_id", ""))
        key = (source_dataset, source_id)
        source = source_index.get(key)
        if source is None:
            raise KeyError(f"원본 manifest에서 레코드를 찾을 수 없습니다: {key}")

        exported_components = [
            component
            for component in record.get("component_audit", [])
            if component.get("status") == "exported"
        ]
        first_label = _resolve(repository_root, export_root / record["exports"][0]["label_path"])
        lines = [line for line in first_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != len(exported_components):
            raise ValueError(
                f"라벨 줄 수와 내보낸 컴포넌트 수가 다릅니다({source_id}): "
                f"{len(lines)} != {len(exported_components)}"
            )

        # 다수결이 필요한 데이터셋은 원본 라벨과 인스턴스 PNG를 한 번만 읽어
        # 모든 컴포넌트에 재사용한다.
        raw_mask: np.ndarray | None = None
        instance_mask: np.ndarray | None = None
        if source_dataset in ("umd", "iit_aff") and lines:
            source_mask_path = _resolve(repository_root, source["source_mask_path"])
            if source_dataset == "umd":
                mat = loadmat(source_mask_path, variable_names=("gt_label",))
                raw_mask = np.asarray(mat["gt_label"]).squeeze()
            else:
                from src.datasets.iit_aff import load_iit_label_matrix

                raw_mask = load_iit_label_matrix(source_mask_path)
            instance_mask = np.asarray(
                Image.open(_resolve(repository_root, source["instance_mask_path"]))
            )
        components_by_id = {
            int(component["component_id"]): component
            for component in source.get("components", [])
        }

        new_lines: list[str] = []
        record_audit: list[dict[str, Any]] = []
        for line, exported in zip(lines, exported_components):
            component_id = int(exported["component_id"])
            old_class = int(exported["model_class_id"])
            component_audit: dict[str, Any] = {
                "component_id": component_id,
                "old_class_id": old_class,
            }
            if old_class == 1:
                # 기존 functional_region은 그대로 3클래스의 2번이 된다.
                new_class: int | None = 2
                component_audit["rule"] = "functional_passthrough"
            elif source_dataset == "custom":
                new_class, info = custom_component_grasp_type(components_by_id[component_id])
                component_audit.update(info)
                component_audit["rule"] = "custom_part_field"
            elif source_dataset in ("umd", "iit_aff"):
                assert raw_mask is not None and instance_mask is not None
                handle_value, body_value = (
                    (UMD_HANDLE_SOURCE_VALUE, UMD_BODY_SOURCE_VALUE)
                    if source_dataset == "umd"
                    else (IIT_HANDLE_SOURCE_VALUE, IIT_BODY_SOURCE_VALUE)
                )
                new_class, info = component_grasp_type_vote(
                    raw_mask, instance_mask, component_id, handle_value, body_value
                )
                component_audit.update(info)
                component_audit["rule"] = f"{source_dataset}_source_vote"
                if component_audit.get("ambiguous"):
                    counters[f"ambiguous_{source_dataset}_components"] += 1
            elif source_dataset == "affgrasp":
                new_class = 0
                component_audit["rule"] = "affgrasp_graspable_assumed_handle"
                counters["affgrasp_assumed_handle"] += 1
            else:
                new_class = None
                component_audit["reason"] = f"unknown_source_dataset:{source_dataset}"

            if new_class is None:
                counters["components_dropped"] += 1
                counters[f"drop_reason:{component_audit.get('reason', 'unknown')}"] += 1
                component_audit["new_class_id"] = None
                record_audit.append(component_audit)
                continue
            component_audit["new_class_id"] = new_class
            class_instances[GRASP_TYPE_NAMES[new_class]] += 1
            new_lines.append(remap_label_line(line, new_class))
            record_audit.append(component_audit)

        label_text = "\n".join(new_lines) + ("\n" if new_lines else "")
        for export in record["exports"]:
            _resolve(repository_root, export_root / export["label_path"]).write_text(
                label_text, encoding="utf-8"
            )
        counters["records_remapped"] += 1
        audit_records.append(
            {
                "source_dataset": source_dataset,
                "source_id": source_id,
                "components": record_audit,
            }
        )

    # dataset.yaml과 버전 기록을 3클래스로 갱신한다.
    dataset_yaml = export_root / "dataset.yaml"
    document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    document["names"] = GRASP_TYPE_NAMES
    dataset_yaml.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    version_path = export_root / "dataset_version.json"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    version["grasp_type_remap"] = {
        "classes": {str(key): value for key, value in GRASP_TYPE_NAMES.items()},
        "umd_rule": "gt_label grasp(1) 대 wrap-grasp(7) 다수결",
        "custom_rule": "승인 manifest part 필드(handle/body)",
        "affgrasp_rule": "graspable을 handle_grasp로 간주(표본 점검 필요)",
        "counts": dict(sorted(counters.items())),
        "class_instances": dict(sorted(class_instances.items())),
    }
    version_path.write_text(
        json.dumps(version, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (export_root / "grasp_type_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "counts": dict(sorted(counters.items())),
        "class_instances": dict(sorted(class_instances.items())),
    }
