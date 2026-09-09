"""파지 후보 선택 패키지.

분할 모델의 영역 출력에서 파지 후보를 추출하고, 손의 위치·접근 방향과
인간 파지 사전확률을 반영한 점수로 최적 후보와 파지 자세
(PRECISION/WRAP/POWER)를 결정한다.
"""

from .candidates import (
    GraspCandidate,
    extract_candidates,
    extract_candidates_from_arrays,
)
from .hand_filter import HandSuppressionFilter
from .scoring import Decision, HandState, decide_grasp, load_selection_config

__all__ = [
    "GraspCandidate",
    "extract_candidates",
    "HandSuppressionFilter",
    "extract_candidates_from_arrays",
    "HandState",
    "Decision",
    "decide_grasp",
    "load_selection_config",
]
