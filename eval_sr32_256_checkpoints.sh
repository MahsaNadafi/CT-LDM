#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

exp="logs/2026-03-05T13-16-36_ldm_sr_16_32x32x32_256"
checkpoints=(
  "epoch=000058.ckpt"
  "epoch=000149.ckpt"
  "epoch=000181.ckpt"
)
steps_list=(100 200 500 1000)

log_dir="$exp/eval_logs"
mkdir -p "$log_dir"

for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint_path="$exp/checkpoints/$checkpoint_name"
  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Missing checkpoint: $checkpoint_path" >&2
    exit 1
  fi

  checkpoint_tag="${checkpoint_name%.ckpt}"
  for steps in "${steps_list[@]}"; do
    tag="${checkpoint_tag}_steps${steps}"
    log_path="$log_dir/eval_sr32_256_${tag}_testdata.log"

    echo "Evaluating $checkpoint_name with $steps DDIM steps"
    python eval_sr.py \
      --exp "$exp" \
      --ckpt "$checkpoint_path" \
      --output_tag "$tag" \
      --split test \
      --lr_size 32 \
      --scale_ratio 8 \
      --first_k -1 \
      --batch_size 1 \
      --steps "$steps" \
      --eta 1.0 \
      --verbose true \
      --save_image false \
      --save_metrics false > "$log_path" 2>&1
  done
done
