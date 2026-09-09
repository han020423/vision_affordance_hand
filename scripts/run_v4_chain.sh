#!/bin/bash
# v4(라벨 충돌 제거) 실험 J + 3-seed 반복 야간 체인
# tmux에서 실행: bash scripts/run_v4_chain.sh
set -o pipefail
cd "$HOME/hjh_vision_hand" || exit 1
# 저장소 루트는 계정마다 다르므로 환경변수로 받는다(미지정 시 /DATA/$USER 아래로 가정).
STORAGE_ROOT="${STORAGE_ROOT:-/DATA/$USER/hjh_vision_hand_storage}"
POLICY_MANIFEST="$STORAGE_ROOT/evaluation_assets/container_policy_audit_test/manifest.jsonl"

echo "=== [1/5] 혼합 v4 생성 ==="
python scripts/build_mixed_finetune_dataset.py \
  --custom data/processed/yolo_grasp_type_custom_v2 \
  --public data/processed/yolo_grasp_type_public_v4 \
  --output data/processed/yolo_mixed_grasp_type_v4 --custom-repeat 10 | tail -n 2 \
  || { echo "CHAIN_FAIL=mix"; exit 1; }

echo "=== [2/5] v4 사전학습 (smoke → 전체) ==="
python scripts/train_seg.py --config configs/training/public_pretrain_grasp_type_s_v4.yaml --smoke-test \
  > /tmp/v4_pre_smoke.log 2>&1 || { echo "CHAIN_FAIL=pre_smoke"; tail -n 5 /tmp/v4_pre_smoke.log; exit 1; }
python scripts/train_seg.py --config configs/training/public_pretrain_grasp_type_s_v4.yaml \
  || { echo "CHAIN_FAIL=pretrain"; exit 1; }

echo "=== [3/5] 실험 J 미세조정 (smoke → 전체) ==="
python scripts/train_seg.py --config configs/training/custom_finetune_j_grasp_type_v4.yaml --smoke-test \
  > /tmp/j_smoke.log 2>&1 || { echo "CHAIN_FAIL=j_smoke"; tail -n 5 /tmp/j_smoke.log; exit 1; }
python scripts/train_seg.py --config configs/training/custom_finetune_j_grasp_type_v4.yaml \
  || { echo "CHAIN_FAIL=j_train"; exit 1; }

echo "=== [4/5] 평가 (mAP + IoU): J와 I ==="
python scripts/evaluate_grasp_type_model.py \
  --model outputs/training/custom_finetune_j_grasp_type_v4_seed42/weights/best.pt \
  --public-data data/processed/yolo_grasp_type_public_v4/dataset.yaml \
  --custom-data data/processed/yolo_grasp_type_custom_v2/dataset.yaml \
  --policy-manifest "$POLICY_MANIFEST" \
  --output-dir outputs/evaluation/custom_finetune_j_v4_eval 2>&1 | tail -n 14 \
  || echo "CHAIN_WARN=j_eval"
python scripts/evaluate_grasp_type_model.py \
  --model outputs/training/custom_finetune_i_grasp_type_v3_seed42/weights/best.pt \
  --public-data data/processed/yolo_grasp_type_public_v3/dataset.yaml \
  --custom-data data/processed/yolo_grasp_type_custom_v2/dataset.yaml \
  --output-dir outputs/evaluation/custom_finetune_i_v3_eval_iou 2>&1 | tail -n 14 \
  || echo "CHAIN_WARN=i_eval"

echo "=== [5/5] 3-seed 반복 (H·I × seed 43/44) ==="
for cfg in custom_finetune_h_grasp_type_v2_seed43 custom_finetune_h_grasp_type_v2_seed44 \
           custom_finetune_i_grasp_type_v3_seed43 custom_finetune_i_grasp_type_v3_seed44; do
  echo "--- $cfg ---"
  python scripts/train_seg.py --config "configs/training/$cfg.yaml" \
    || { echo "CHAIN_FAIL=$cfg"; exit 1; }
done

echo "CHAIN_V4_DONE=0"
