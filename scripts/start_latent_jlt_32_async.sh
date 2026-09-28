#!/bin/bash
# JLT on ImageNet, FLUX.2 latent, image 256 -> 16x16 latent, patch 2 (JLT-B/2).
# Each token covers a 32x32 raw-pixel receptive field (16x VAE compression * 2 patch).
#
# Usage:
#   ./scripts/start_latent_jlt_32_async.sh              # default: all 8 GPUs
#   ./scripts/start_latent_jlt_32_async.sh 0,1,2,3      # subset
set -euo pipefail

# Ensure execution from repository root
cd "$(dirname "$0")/.."

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN=latent-JLT-B-2-Async-teacher
OUTPUT_DIR=/mnt/raid0/JLT/output_dir/${RUN}
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port="${MASTER_PORT:-29500}" \
    main_jlt.py \
        --model JLT-B/2 \
        --vae_type flux2 \
        --proj_dropout 0.0 \
        --P_mean -0.8 --P_std 0.8 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 512 --blr 5e-5 \
        --epochs 40 --warmup_epochs 5 \
        --gen_bsz 128 --num_images 50000 \
        --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
        --eval_freq 10 --online_eval  --async_timesteps --async_timestep_drop 0.1 --ema_feat_align_weight 0.8 --ema_feat_align_teacher_layers 9 --ema_feat_align_student_layers 4 \
        --num_workers 12 \
        --data_path /mnt/raid0/JLT/data/imagenet_latents_256 --use_latent_cache \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project ciallo --wandb_name "${RUN}" --wandb_mode online
