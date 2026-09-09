#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "사용법: bash scripts/setup_training_server.sh <pytorch-wheel-index-url>" >&2
  echo "서버 드라이버에 맞는 wheel 주소를 pytorch.org에서 선택해야 합니다." >&2
  exit 2
fi

# 프로젝트 전용 가상환경에만 패키지를 설치하며 서버 전체 환경은 변경하지 않는다.
VISION_HAND_TORCH_INDEX_URL="$1"
python3 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$VISION_HAND_TORCH_INDEX_URL"
python -m pip install -r requirements-server.txt
python scripts/check_training_server.py
