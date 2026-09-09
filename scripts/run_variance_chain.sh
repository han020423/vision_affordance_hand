#!/bin/bash
# 실행 분산 측정 체인: 데이터 순서 재배열(ord 43/44)로 H·I 반복 학습
# (Ultralytics 8.3.163은 seed가 데이터로더에 반영되지 않아 완전 결정적이므로,
#  train 목록 순서 재배열로 무작위성을 만든다. ord42 = 기존 결정적 순서 실행)
set -o pipefail
cd "$HOME/hjh_vision_hand" || exit 1

echo "=== [1/2] ord 혼합 데이터셋 4개 생성 ==="
for pair in "v2 43" "v2 44" "v3 43" "v3 44"; do
  set -- $pair
  ver=$1; seed=$2
  python scripts/build_mixed_finetune_dataset.py \
    --custom data/processed/yolo_grasp_type_custom_v2 \
    --public data/processed/yolo_grasp_type_public_$([ "$ver" = "v2" ] && echo v2 || echo v3) \
    --output "data/processed/yolo_mixed_grasp_type_${ver}_ord${seed}" \
    --custom-repeat 10 --order-seed "$seed" | tail -n 1 \
    || { echo "VAR_FAIL=mix_${ver}_${seed}"; exit 1; }
done

echo "=== [2/2] 4회 학습 ==="
for cfg in custom_finetune_h_grasp_type_v2_ord43 custom_finetune_h_grasp_type_v2_ord44 \
           custom_finetune_i_grasp_type_v3_ord43 custom_finetune_i_grasp_type_v3_ord44; do
  echo "--- $cfg ---"
  python scripts/train_seg.py --config "configs/training/$cfg.yaml" \
    || { echo "VAR_FAIL=$cfg"; exit 1; }
done

echo "VAR_CHAIN_DONE=0"
