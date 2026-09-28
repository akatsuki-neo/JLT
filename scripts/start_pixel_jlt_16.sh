#!/bin/bash
# JLT on ImageNet, pixel-space, image 256, patch 16 (JLT-B/16 recipe).
#
# Usage:
#   ./scripts/start_pixel_jlt_16.sh              # default: all 8 GPUs
#   ./scripts/start_pixel_jlt_16.sh 0,1,2,3      # subset
set -euo pipefail

# Ensure execution from repository root
cd "$(dirname "$0")/.."

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN=pixel-JLT-B-16
OUTPUT_DIR=/mnt/raid0/JLT/output_dir/${RUN}
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port="${MASTER_PORT:-29500}" \
    main_jlt.py \
        --model JLT-B/16 \
        --vae_type identity \
        --proj_dropout 0.0 \
        --P_mean -0.8 --P_std 0.8 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 128 --accum_iter 4 --blr 5e-5 \
        --epochs 40 --warmup_epochs 5 \
        --gen_bsz 128 --num_images 50000 \
        --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
        --eval_freq 10 --online_eval \
        --num_workers 12 \
        --data_path /mnt/raid0/LightningDiT/data --use_parquet \
        --cache_dir /mnt/raid0/JLT/hf_cache \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project ciallo --wandb_name "${RUN}" --wandb_mode online
