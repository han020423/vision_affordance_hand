"""OpenCV 화면에서 머그 손잡이·몸통 후보 마스크를 직접 수정한다.

원본 반자동 후보는 절대 덮어쓰지 않는다. 이 스크립트는
``outputs/custom_mask_review`` 아래의 작업 복사본만 변경한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.labeling.custom_review import (  # noqa: E402
    BODY_INSTANCE_ID,
    FUNCTIONAL_INSTANCE_ID,
    HANDLE_INSTANCE_ID,
    render_review_overlay,
    semantic_from_instances,
    validate_review_masks,
)


HEADER_HEIGHT = 46


def parse_args() -> argparse.Namespace:
    """수정 작업 폴더와 화면 크기를 명령행에서 받을 수 있게 한다."""

    parser = argparse.ArgumentParser(
        description="손잡이와 몸통 마스크를 브러시·다각형으로 수정합니다."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/custom_mask_review/review_20260818"),
        help="prepare_custom_mask_review.py가 만든 수정 작업 폴더",
    )
    parser.add_argument(
        "--start-file",
        help="특정 검수 파일명부터 시작합니다.",
    )
    parser.add_argument("--max-width", type=int, default=1400, help="최대 화면 너비")
    parser.add_argument("--max-height", type=int, default=900, help="최대 화면 높이")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """검수 진행 상태가 저장된 JSONL을 읽는다."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    """중간 종료에도 manifest가 깨지지 않도록 임시 파일을 거쳐 교체한다."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_image_required(path: Path, image: np.ndarray) -> None:
    """마스크나 오버레이 저장 실패를 즉시 사용자에게 알린다."""

    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")


@dataclass
class EditorState:
    """현재 이미지와 마우스 수정 상태를 한곳에 보관한다."""

    workspace: Path
    manifest_path: Path
    rows: list[dict[str, object]]
    index: int
    max_width: int
    max_height: int
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    candidate_mask: np.ndarray | None = None
    selected_id: int = HANDLE_INSTANCE_ID
    brush_radius: int = 18
    mode: str = "brush"
    polygon_points: list[tuple[int, int]] = field(default_factory=list)
    history: list[np.ndarray] = field(default_factory=list)
    drawing: bool = False
    dirty: bool = False
    handle_not_visible: bool = False
    scale: float = 1.0
    exclude_armed: bool = False
    quit_armed: bool = False

    @property
    def row(self) -> dict[str, object]:
        """현재 manifest 행을 반환한다."""

        return self.rows[self.index]

    def _read_required_image(self, path: Path, mode: int) -> np.ndarray:
        """필수 이미지를 읽고 실패 시 파일명을 포함해 중단한다."""

        image = cv2.imread(str(path), mode)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
        return image

    def load_current(self) -> None:
        """현재 행의 원본과 수정 복사본을 불러온다."""

        self.image = self._read_required_image(
            self.workspace / str(self.row["image_path"]), cv2.IMREAD_COLOR
        )
        self.mask = self._read_required_image(
            self.workspace / str(self.row["corrected_instance_path"]),
            cv2.IMREAD_UNCHANGED,
        ).astype(np.uint8)
        self.candidate_mask = self._read_required_image(
            self.workspace / str(self.row["candidate_instance_path"]),
            cv2.IMREAD_UNCHANGED,
        ).astype(np.uint8)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError(f"이미지와 마스크 크기가 다릅니다: {self.row['source_id']}")
        self.polygon_points.clear()
        self.history.clear()
        self.dirty = False
        self.handle_not_visible = self.row.get("handle_visibility") == "not_visible"
        self.exclude_armed = False
        self.quit_armed = False
        print(
            f"\n[{self.index + 1}/{len(self.rows)}] {self.row['source_id']}\n"
            f"검수 의견: {self.row['review_note']}\n"
            f"현재 상태: {self.row['review_status']}"
        )

    def push_history(self) -> None:
        """실수한 한 번의 동작을 되돌릴 수 있도록 이전 마스크를 저장한다."""

        assert self.mask is not None
        self.history.append(self.mask.copy())
        if len(self.history) > 20:
            self.history.pop(0)

    def paint(self, original_x: int, original_y: int, value: int) -> None:
        """원본 해상도 좌표에 선택한 부위 ID를 원형 브러시로 칠한다."""

        assert self.mask is not None
        cv2.circle(
            self.mask,
            (original_x, original_y),
            self.brush_radius,
            int(value),
            thickness=-1,
        )
        self.dirty = True

    def save_approved(self) -> bool:
        """카테고리 규칙을 만족할 때만 수정 마스크를 승인 저장한다."""

        assert self.mask is not None and self.image is not None
        handle_pixels = int(np.count_nonzero(self.mask == HANDLE_INSTANCE_ID))
        body_pixels = int(np.count_nonzero(self.mask == BODY_INSTANCE_ID))
        functional_pixels = int(np.count_nonzero(self.mask == FUNCTIONAL_INSTANCE_ID))
        # 과거 워크스페이스(머그 전용)에는 category 필드가 없으므로 mug로 간주한다.
        category = str(self.row.get("category") or "mug")
        problems = validate_review_masks(
            category,
            handle_pixels=handle_pixels,
            body_pixels=body_pixels,
            functional_pixels=functional_pixels,
            handle_not_visible=self.handle_not_visible,
        )
        if problems:
            print("저장하지 않았습니다:")
            for problem in problems:
                print(" -", problem)
            print("판단할 수 없는 사진은 X를 두 번 눌러 제외하세요.")
            return False

        corrected_path = self.workspace / str(self.row["corrected_instance_path"])
        semantic_path = self.workspace / "corrected_semantic" / corrected_path.name
        overlay_path = self.workspace / "corrected_overlays" / corrected_path.with_suffix(
            ".jpg"
        ).name
        write_image_required(corrected_path, self.mask)
        write_image_required(semantic_path, semantic_from_instances(self.mask))
        write_image_required(overlay_path, render_review_overlay(self.image, self.mask))
        self.row.update(
            {
                "review_status": "human_corrected",
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "corrected_semantic_path": semantic_path.relative_to(
                    self.workspace
                ).as_posix(),
                "corrected_overlay_path": overlay_path.relative_to(
                    self.workspace
                ).as_posix(),
                "handle_pixels": handle_pixels,
                "body_pixels": body_pixels,
                "functional_pixels": functional_pixels,
                "handle_visibility": (
                    "not_visible" if self.handle_not_visible else "visible"
                ),
                "functional_region_generated": functional_pixels > 0,
            }
        )
        write_jsonl_atomic(self.manifest_path, self.rows)
        self.dirty = False
        print("수정 마스크를 승인 저장했습니다.")
        return True

    def mark_excluded(self) -> None:
        """가림 등으로 정답을 판단할 수 없는 이미지를 학습 제외로 기록한다."""

        self.row.update(
            {
                "review_status": "excluded_ambiguous",
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "functional_region_generated": False,
            }
        )
        write_jsonl_atomic(self.manifest_path, self.rows)
        self.dirty = False
        self.exclude_armed = False
        print("판단 불가능 항목으로 제외했습니다. 원본과 후보 파일은 보존됩니다.")

    def move(self, offset: int) -> None:
        """수정 내용을 버리지 않도록 확인한 뒤 앞뒤 항목으로 이동한다."""

        if self.dirty:
            print("저장하지 않은 수정이 있습니다. A로 저장하거나 R로 초기화하세요.")
            return
        self.index = (self.index + offset) % len(self.rows)
        self.save_session_position()
        self.load_current()

    def save_session_position(self) -> None:
        """도구를 다시 열 때 마지막으로 보던 이미지부터 이어서 시작하게 한다."""

        session_path = self.workspace / "editor_session.json"
        session_path.write_text(
            json.dumps(
                {
                    "last_index": self.index,
                    "last_source_id": self.row["source_id"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def display_image(self) -> np.ndarray:
        """현재 마스크·다각형·단축키를 포함한 화면 이미지를 만든다."""

        assert self.image is not None and self.mask is not None
        overlay = render_review_overlay(self.image, self.mask)
        if self.polygon_points:
            points = np.asarray(self.polygon_points, dtype=np.int32)
            cv2.polylines(overlay, [points], False, (0, 0, 255), 3)
            for point in self.polygon_points:
                cv2.circle(overlay, point, 5, (0, 0, 255), -1)

        height, width = overlay.shape[:2]
        available_height = max(100, self.max_height - HEADER_HEIGHT)
        self.scale = min(1.0, self.max_width / width, available_height / height)
        if self.scale != 1.0:
            overlay = cv2.resize(
                overlay,
                (round(width * self.scale), round(height * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        header = np.full((HEADER_HEIGHT, overlay.shape[1], 3), 35, dtype=np.uint8)
        selected = {
            0: "ERASE",
            HANDLE_INSTANCE_ID: "HANDLE",
            BODY_INSTANCE_ID: "BODY",
            FUNCTIONAL_INSTANCE_ID: "FUNCTIONAL",
        }[self.selected_id]
        status = str(self.row["review_status"])
        visibility = "HIDDEN" if self.handle_not_visible else "VISIBLE"
        category = str(self.row.get("category") or "mug")
        first = (
            f"{self.index + 1}/{len(self.rows)}  {self.row['source_id']}  [{category}]  "
            f"selected={selected} mode={self.mode} brush={self.brush_radius} handle={visibility}"
        )
        second = f"status={status} | H handle, B body, F functional, E erase, V hidden, M mode, A save, N/P move, Q quit"
        cv2.putText(header, first, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(header, second, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
        return np.vstack([header, overlay])


def mouse_callback(event: int, x: int, y: int, flags: int, state: EditorState) -> None:
    """화면 좌표를 원본 좌표로 바꿔 브러시와 다각형 입력을 처리한다."""

    if state.image is None or y < HEADER_HEIGHT:
        return
    original_x = int(round(x / state.scale))
    original_y = int(round((y - HEADER_HEIGHT) / state.scale))
    height, width = state.image.shape[:2]
    original_x = max(0, min(width - 1, original_x))
    original_y = max(0, min(height - 1, original_y))

    if state.mode == "polygon":
        if event == cv2.EVENT_LBUTTONDOWN:
            state.polygon_points.append((original_x, original_y))
        elif event == cv2.EVENT_RBUTTONDOWN and state.polygon_points:
            state.polygon_points.pop()
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.push_history()
        state.drawing = True
        state.paint(original_x, original_y, state.selected_id)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.push_history()
        state.drawing = True
        state.paint(original_x, original_y, 0)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        value = 0 if flags & cv2.EVENT_FLAG_RBUTTON else state.selected_id
        state.paint(original_x, original_y, value)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state.drawing = False


def fill_polygon(state: EditorState) -> None:
    """세 점 이상 찍은 다각형을 선택 영역으로 채운다."""

    if len(state.polygon_points) < 3:
        print("다각형은 점을 세 개 이상 찍어야 합니다.")
        return
    assert state.mask is not None
    state.push_history()
    points = np.asarray(state.polygon_points, dtype=np.int32)
    cv2.fillPoly(state.mask, [points], int(state.selected_id))
    state.polygon_points.clear()
    state.dirty = True


def first_start_index(
    rows: list[dict[str, object]],
    start_file: str | None,
    workspace: Path,
) -> int:
    """지정 파일 또는 아직 수정하지 않은 첫 항목 위치를 찾는다."""

    if start_file:
        for index, row in enumerate(rows):
            if Path(str(row["image_path"])).name == start_file:
                return index
        raise ValueError(f"검수 manifest에 없는 시작 파일입니다: {start_file}")
    session_path = workspace / "editor_session.json"
    if session_path.is_file():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        last_source_id = session.get("last_source_id")
        for index, row in enumerate(rows):
            if row.get("source_id") == last_source_id:
                return index
    for index, row in enumerate(rows):
        if row.get("review_status") == "pending_correction":
            return index
    return 0


def main() -> int:
    """검수 창을 열고 키 입력에 따라 안전하게 수정본을 저장한다."""

    args = parse_args()
    workspace = args.workspace if args.workspace.is_absolute() else REPOSITORY_ROOT / args.workspace
    manifest_path = workspace / "review_manifest.jsonl"
    rows = load_jsonl(manifest_path)
    if not rows:
        raise ValueError("검수할 항목이 없습니다")
    state = EditorState(
        workspace=workspace,
        manifest_path=manifest_path,
        rows=rows,
        index=first_start_index(rows, args.start_file, workspace),
        max_width=args.max_width,
        max_height=args.max_height,
    )
    state.load_current()

    window_name = "Custom mug mask review"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback, state)
    print(
        "단축키: H 손잡이(초록), B 몸통(청록, 머그만), F 기능 부위(빨강, 가위 날·드라이버 축), "
        "E 지우기, M 브러시/다각형 전환, Enter 다각형 채우기, [/] 브러시 크기, U 되돌리기, "
        "V 손잡이 비가시, R 후보로 초기화, A 승인 저장, N/P 이동, X 두 번 제외, Q 종료"
    )

    while True:
        cv2.imshow(window_name, state.display_image())
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        # X/Q 이중 확인은 다른 키를 누르는 즉시 취소해 오조작을 막는다.
        if key not in (ord("x"), ord("X")):
            state.exclude_armed = False
        if key not in (ord("q"), ord("Q")):
            state.quit_armed = False
        if key in (ord("h"), ord("H")):
            state.selected_id = HANDLE_INSTANCE_ID
        elif key in (ord("b"), ord("B")):
            state.selected_id = BODY_INSTANCE_ID
        elif key in (ord("f"), ord("F")):
            state.selected_id = FUNCTIONAL_INSTANCE_ID
        elif key in (ord("e"), ord("E")):
            state.selected_id = 0
        elif key in (ord("v"), ord("V")):
            state.handle_not_visible = not state.handle_not_visible
            state.dirty = True
            print(
                "손잡이 비가시 상태:",
                "사용" if state.handle_not_visible else "해제",
            )
        elif key in (ord("m"), ord("M")):
            state.mode = "polygon" if state.mode == "brush" else "brush"
            state.polygon_points.clear()
        elif key in (10, 13):
            fill_polygon(state)
        elif key == 27:
            state.polygon_points.clear()
        elif key == ord("["):
            state.brush_radius = max(2, state.brush_radius - 3)
        elif key == ord("]"):
            state.brush_radius = min(200, state.brush_radius + 3)
        elif key in (ord("u"), ord("U")) and state.history:
            state.mask = state.history.pop()
            state.dirty = True
        elif key in (ord("r"), ord("R")):
            assert state.candidate_mask is not None
            state.push_history()
            state.mask = state.candidate_mask.copy()
            state.dirty = True
            state.polygon_points.clear()
            print("현재 항목을 원본 후보 상태로 되돌렸습니다. A를 눌러 저장하세요.")
        elif key in (ord("a"), ord("A")):
            if state.save_approved():
                state.move(1)
        elif key in (ord("n"), ord("N")):
            state.move(1)
        elif key in (ord("p"), ord("P")):
            state.move(-1)
        elif key in (ord("x"), ord("X")):
            if state.exclude_armed:
                state.mark_excluded()
                state.move(1)
            else:
                state.exclude_armed = True
                print("정말 제외하려면 X를 한 번 더 누르세요.")
        elif key in (ord("q"), ord("Q")):
            if state.dirty and not state.quit_armed:
                state.quit_armed = True
                print("저장하지 않은 수정이 있습니다. 버리고 종료하려면 Q를 다시 누르세요.")
            else:
                state.save_session_position()
                break

    cv2.destroyAllWindows()
    statuses: dict[str, int] = {}
    for row in state.rows:
        status = str(row["review_status"])
        statuses[status] = statuses.get(status, 0) + 1
    print("현재 검수 상태:", json.dumps(statuses, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
