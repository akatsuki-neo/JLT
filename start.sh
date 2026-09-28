#!/bin/bash
# One-shot JiT training launcher on ImageNet (parquet).
#
# Usage:
#   ./start.sh              # default: all 8 GPUs
#   ./start.sh 0,1,2,3      # subset
set -euo pipefail

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

OUTPUT_DIR=/path/to/jlt/output_dir/JiT-B-16-256
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    main_jit.py \
        --model JiT-B/16 \
        --proj_dropout 0.0 \
        --P_mean -0.8 --P_std 0.8 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 256 --blr 5e-5 \
        --epochs 40 --warmup_epochs 5 \
        --gen_bsz 128 --num_images 50000 \
        --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
        --eval_freq 40 --online_eval \
        --num_workers 12 \
        --data_path /path/to/imagenet --use_parquet \
        --cache_dir /path/to/jlt/hf_cache \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project jlt --wandb_name JiT-B-16-256 --wandb_mode online
