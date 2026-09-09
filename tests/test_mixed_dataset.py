"""혼합 replay 데이터셋 생성 스크립트의 검증 로직 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_mixed_finetune_dataset import (  # noqa: E402
    build_mixed_dataset,
    ensure_no_split_overlap,
)

EXPECTED_NAMES = {0: "grasp_region", 1: "functional_region"}


def make_source_dataset(
    root: Path,
    train_names: list[str],
    val_names: list[str],
    names: dict[int, str] | None = None,
    empty_label_names: set[str] | None = None,
) -> None:
    """최소 크기의 합성 YOLO 데이터셋 구조를 만든다. 파서 검증 전용이다."""

    empty_label_names = empty_label_names or set()
    for split, sample_names in (("train", train_names), ("val", val_names)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for name in sample_names:
            (image_dir / f"{name}.jpg").write_bytes(b"\xff\xd8fake")
            if name in empty_label_names:
                (label_dir / f"{name}.txt").write_text("", encoding="utf-8")
            else:
                (label_dir / f"{name}.txt").write_text(
                    "0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8"
                )
    with (root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": names if names is not None else EXPECTED_NAMES,
            },
            handle,
        )


class TestEnsureNoSplitOverlap(unittest.TestCase):
    def test_중복이_없으면_통과한다(self) -> None:
        ensure_no_split_overlap([Path("/a/1.jpg")], [Path("/a/2.jpg")])

    def test_같은_이미지가_양쪽에_있으면_실패한다(self) -> None:
        shared = Path("/a/1.jpg")
        with self.assertRaises(ValueError):
            ensure_no_split_overlap([shared, Path("/a/2.jpg")], [shared])


class TestBuildMixedDataset(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.base = Path(self._temp.name)
        self.custom = self.base / "custom"
        self.public = self.base / "public"
        self.output = self.base / "mixed"
        # 자체 데이터에는 배경 음성 샘플의 빈 라벨도 포함시켜 실제 구성을 흉내낸다.
        make_source_dataset(
            self.custom,
            ["mug_a", "mug_b", "bg_a"],
            ["mug_val"],
            empty_label_names={"bg_a"},
        )
        make_source_dataset(self.public, ["tool_a", "tool_b"], ["tool_val"])

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_반복_배수와_수량이_기록된다(self) -> None:
        manifest = build_mixed_dataset(self.custom, self.public, self.output, custom_repeat=4)
        counts = manifest["counts"]
        self.assertEqual(counts["custom_train_unique"], 3)
        self.assertEqual(counts["public_train"], 2)
        self.assertEqual(counts["train_entries_with_repeats"], 3 * 4 + 2)
        self.assertEqual(counts["val_entries"], 2)
        train_lines = (self.output / "train.txt").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(train_lines), 14)
        # 자체 이미지가 정확히 4번씩 반복되는지 확인한다.
        self.assertEqual(sum(1 for line in train_lines if line.endswith("mug_a.jpg")), 4)
        self.assertEqual(sum(1 for line in train_lines if line.endswith("tool_a.jpg")), 1)
        document = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(document["names"], EXPECTED_NAMES)
        saved = json.loads((self.output / "mixed_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["custom_repeat"], 4)
        self.assertFalse(saved["source_datasets_modified"])

    def test_클래스_매핑이_다르면_실패한다(self) -> None:
        wrong = self.base / "wrong"
        make_source_dataset(wrong, ["x"], ["y"], names={0: "grasp_region", 1: "contain"})
        with self.assertRaises(ValueError):
            build_mixed_dataset(self.custom, wrong, self.output, custom_repeat=1)

    def test_라벨이_없으면_실패한다(self) -> None:
        (self.public / "labels" / "train" / "tool_a.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            build_mixed_dataset(self.custom, self.public, self.output, custom_repeat=1)

    def test_기존_출력_폴더를_덮어쓰지_않는다(self) -> None:
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            build_mixed_dataset(self.custom, self.public, self.output, custom_repeat=1)

    def test_반복_배수가_0이면_실패한다(self) -> None:
        with self.assertRaises(ValueError):
            build_mixed_dataset(self.custom, self.public, self.output, custom_repeat=0)


if __name__ == "__main__":
    unittest.main()
