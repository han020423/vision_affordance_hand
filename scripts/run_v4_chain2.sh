#!/bin/bash
# v4 체인 2부: 평가(mAP+IoU) 재시도 + 3-seed 반복 (1부에서 실패한 구간)
set -o pipefail
cd "$HOME/hjh_vision_hand" || exit 1
# 저장소 루트는 계정마다 다르므로 환경변수로 받는다(미지정 시 /DATA/$USER 아래로 가정).
STORAGE_ROOT="${STORAGE_ROOT:-/DATA/$USER/hjh_vision_hand_storage}"
POLICY_MANIFEST="$STORAGE_ROOT/evaluation_assets/container_policy_audit_test/manifest.jsonl"

echo "=== [1/2] 평가 (mAP + IoU): J와 I ==="
rm -rf outputs/evaluation/custom_finetune_j_v4_eval outputs/evaluation/custom_finetune_i_v3_eval_iou
python scripts/evaluate_grasp_type_model.py \
  --model outputs/training/custom_finetune_j_grasp_type_v4_seed42/weights/best.pt \
  --public-data data/processed/yolo_grasp_type_public_v4/dataset.yaml \
  --custom-data data/processed/yolo_grasp_type_custom_v2/dataset.yaml \
  --policy-manifest "$POLICY_MANIFEST" \
  --output-dir outputs/evaluation/custom_finetune_j_v4_eval 2>&1 | tail -n 16 \
  || { echo "CHAIN2_FAIL=j_eval"; exit 1; }
python scripts/evaluate_grasp_type_model.py \
  --model outputs/training/custom_finetune_i_grasp_type_v3_seed42/weights/best.pt \
  --public-data data/processed/yolo_grasp_type_public_v3/dataset.yaml \
  --custom-data data/processed/yolo_grasp_type_custom_v2/dataset.yaml \
  --output-dir outputs/evaluation/custom_finetune_i_v3_eval_iou 2>&1 | tail -n 16 \
  || { echo "CHAIN2_FAIL=i_eval"; exit 1; }

echo "=== [2/2] 3-seed 반복 (H·I × seed 43/44) ==="
for cfg in custom_finetune_h_grasp_type_v2_seed43 custom_finetune_h_grasp_type_v2_seed44 \
           custom_finetune_i_grasp_type_v3_seed43 custom_finetune_i_grasp_type_v3_seed44; do
  echo "--- $cfg ---"
  python scripts/train_seg.py --config "configs/training/$cfg.yaml" \
    || { echo "CHAIN2_FAIL=$cfg"; exit 1; }
done

echo "CHAIN2_DONE=0"
