#!/bin/bash
# JiT on ImageNet, FLUX.2 latent, image 256 -> 16x16 latent, patch 2 (JiT-B/2).
# Each token covers a 32x32 raw-pixel receptive field (16x VAE compression * 2 patch).
#
# Usage:
#   ./start_latent_jit_32.sh              # default: all 8 GPUs
#   ./start_latent_jit_32.sh 0,1,2,3      # subset
set -euo pipefail

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN=latent-JiT-B-2
OUTPUT_DIR=/path/to/jlt/output_dir/${RUN}
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port="${MASTER_PORT:-29500}" \
    main_jit.py \
        --model JiT-B/2 \
        --vae_type flux2 \
        --proj_dropout 0.0 \
        --P_mean -0.8 --P_std 0.8 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 256 --blr 5e-5 \
        --epochs 200 --warmup_epochs 5 \
        --gen_bsz 256 --num_images 50000 \
        --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
        --eval_freq 40 --online_eval \
        --num_workers 12 \
        --data_path /path/to/jlt/data/imagenet_latents_256 --use_latent_cache \
        --vae_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project jlt --wandb_name "${RUN}" --wandb_mode online
