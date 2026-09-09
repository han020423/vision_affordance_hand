"""통합 affordance 마스크의 연결 성분을 분리된 상태로 보존한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .policy import FUNCTIONAL_STORAGE_VALUE, GRASP_STORAGE_VALUE, MODEL_CLASS_IDS


@dataclass(frozen=True)
class Component:
    """분리된 affordance 후보 하나의 메타데이터."""

    component_id: int
    class_name: str
    model_class_id: int
    storage_value: int
    pixel_count: int
    bbox_xywh: tuple[int, int, int, int]
    touches_boundary: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    """같은 클래스 영역도 서로 분리한 uint16 인스턴스 맵을 반환한다."""

    if mask.ndim != 2:
        raise ValueError(f"2차원 마스크가 필요하지만 실제 크기는 {mask.shape}입니다")

    instance_mask = np.zeros(mask.shape, dtype=np.uint16)
    components: list[Component] = []
    next_id = 1
    height, width = mask.shape

    for storage_value, class_name in (
        (GRASP_STORAGE_VALUE, "grasp_region"),
        (FUNCTIONAL_STORAGE_VALUE, "functional_region"),
    ):
        binary = (mask == storage_value).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for local_id in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[local_id])
            if next_id > np.iinfo(np.uint16).max:
                raise ValueError("uint16 인스턴스 마스크로 표현하기에는 연결 성분이 너무 많습니다")
            instance_mask[labels == local_id] = next_id
            components.append(
                Component(
                    component_id=next_id,
                    class_name=class_name,
                    model_class_id=MODEL_CLASS_IDS[class_name],
                    storage_value=storage_value,
                    pixel_count=area,
                    bbox_xywh=(x, y, w, h),
                    touches_boundary=x == 0 or y == 0 or x + w == width or y + h == height,
                )
            )
            next_id += 1

    return instance_mask, components
