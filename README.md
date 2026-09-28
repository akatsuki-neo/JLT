# JLT: Clean-Latent Prediction in Latent Diffusion Transformers

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2605.27102-b31b1b.svg)](https://arxiv.org/abs/2605.27102)

</div>

## Overview

<div align="center">
<img src="images/jlt_b16_heun50_samples.png" width="75%">
<br><br>
ImageNet 256×256 samples from JLT-B/1 using 50-step Heun sampling.
</div>

## Implementation

### Installation

```bash
# Clone repository
cd JLT

# Create conda environment
conda env create -f environment.yaml
conda activate jit

# Install accelerate (required for distributed training)
pip install accelerate

# Install additional dependencies
pip install torch-fidelity  # for FID evaluation
```

### Data Preparation

#### 1. Download ImageNet

Download ImageNet train/val from [image-net.org](https://image-net.org/download.php) and extract to a directory.

#### 2. Encode Images to FLUX.2 Latents

Encode ImageNet to latent shards for efficient training:

```bash
python prepare_ref.py \
    --data_path /path/to/imagenet \
    --output_path /path/to/imagenet_latents_256 \
    --img_size 256 \
    --vae_type flux2 \
    --vae_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
    --batch_size 256 \
    --num_workers 8
```

This produces safetensor latent shards in `/path/to/imagenet_latents_256`.

### Running Experiments

#### JLT-B/1 (Clean-Latent Prediction, /1 scale)

```bash
./start_latent_jit_16.sh [GPU_IDS]

# Example: use GPUs 0-3 only
./start_latent_jit_16.sh 0,1,2,3
```

Key settings:
- Model: JiT-B/1 (patch 1, 16x16 latent grid)
- Batch size: 256 × 8 GPUs × 2 accum = 4096 effective
- Epochs: 40 (with 5 warmup)
- Learning rate: 5e-5 base LR

#### JLT-B/2 (Clean-Latent Prediction, /2 scale)

```bash
./start_latent_jit_32.sh [GPU_IDS]
```

#### DiT-B/2 Baseline (Velocity Prediction)

```bash
./start_latent_v_32.sh [GPU_IDS]
```

Key difference: `--flow_matching` flag enables direct velocity prediction.

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model variant: `JiT-B/1` or `JiT-B/2` |
| `--vae_type` | `flux2` for FLUX.2 latent space |
| `--flow_matching` | Enable velocity prediction (DiT baseline) |
| `--batch_size` | Micro batch per GPU |
| `--blr` | Base learning rate |
| `--epochs` | Training epochs |
| `--cfg` | Classifier-free guidance scale |
| `--data_path` | Path to pre-encoded latents |
| `--use_latent_cache` | Load pre-encoded safetensor latents |
| `--vae_model_name_or_path` | FLUX.2 VAE path or HuggingFace repo |

### Evaluation

Run evaluation with a trained checkpoint (requires pre-encoded latents):

```bash
python main_jit.py \
    --model JiT-B/1 \
    --vae_type flux2 \
    --img_size 256 \
    --data_path /path/to/imagenet_latents_256 \
    --use_latent_cache \
    --online_eval \
    --eval_freq 1 \
    --gen_bsz 128 \
    --num_images 50000 \
    --cfg 2.9 \
    --num_sampling_steps 50 \
    --resume /path/to/checkpoint-last.pth \
    --output_dir ./eval_output
```

For FID evaluation, pre-encoded ImageNet latents and reference statistics are required. See [torch-fidelity](https://github.com/toshas/torch-fidelity) for details.

## Method

### Formulation and Prediction Targets

Let $x \in \mathbb{R}^D$ denote the clean latent produced by a fixed encoder, and let $\epsilon \sim \mathcal{N}(0, I)$ denote Gaussian noise in the same coordinate system. We use the linear corruption path:

$$z_t = t \cdot x + (1 - t) \cdot \epsilon, \quad t \in [0, 1]$$

The three common direct targets are:

$$y_x = x, \quad y_\epsilon = \epsilon, \quad y_v = x - \epsilon$$

For fixed $t$, $x$-, $\epsilon$-, and $v$-parameterizations are algebraically equivalent: once a model predicts any one target, the other endpoint variables can be recovered by an affine readout from the predicted target and the known mixture $z_t$.

### Target-Geometry Analysis

**Local linear-Gaussian assumption:** $x \sim \mathcal{N}(0, \Sigma)$ with independent noise $\epsilon \sim \mathcal{N}(0, I)$. The marginal target covariances are:

$$
\text{Cov}(y_x) = \Sigma, \quad \text{Cov}(y_\epsilon) = I, \quad \text{Cov}(y_v) = \Sigma + I
$$

**Key insight:** Velocity prediction adds the same isotropic unit floor to every clean-latent direction. If $\Sigma$ is anisotropic, directions with little clean-data variation become unit-variance directions in $y_v$, while clean prediction keeps their target variance small.

**Conditional ambiguity gap:**

$$\frac{\text{Var}(v_i | z_i)}{\text{Var}(x_i | z_i)} = \frac{1}{(1-t)^2} > 1$$

When $\lambda_i \rightarrow 0$ (low-variance directions):

| Prediction Target | Coefficient Tends To |
|-------------------|---------------------|
| Clean ($x$) | $0$ (attenuated) |
| Velocity ($v$) | $-\frac{1}{1-t}$ (amplified) |

## Architecture

JLT is a Base-scale latent Transformer following JiT-B/16 for architectural comparability:

| Component | Specification |
|-----------|--------------|
| Transformer Blocks | 12 |
| Hidden Dimension | 768 |
| Attention Heads | 12 |
| Bottleneck Patch Embedding | 128-dim |
| Parameters | 130M |
| Tokenizer | FLUX.2 VAE (frozen) |

## Experiments

### Matched Target Ablation

<div align="center">

| Model | Target | Guidance | FID-50K ↓ | IS ↑ |
|-------|--------|----------|-----------|------|
| **JLT-B/1** | $x$ (clean) | w/ CFG | **2.56** | 220.74 |
| DiT-B/1 | $v$ (velocity) | w/ CFG | 6.56 | 132.12 |
| **JLT-B/2** | $x$ (clean) | w/ CFG | **14.81** | 107.29 |
| DiT-B/2 | $v$ (velocity) | w/ CFG | 28.71 | 58.46 |
| **JLT-B/1** (final) | $x$ | w/ CFG | **2.50** | **232.51** |
| JLT-B/1 | $x$ | w/o CFG | 14.00 | -- |

*Matched latent target ablation on ImageNet 256×256. The upper block is the controlled target comparison; the lower block reports the selected final JLT-B/1 evaluation.*

</div>

### Comparison with Representative Baselines

<div align="center">

| Model | Space | Params | Train | FID-50K ↓ | IS ↑ |
|-------|-------|--------|-------|-----------|------|
| **JLT-B/1** | FLUX.2 | 130M | 250K/200ep | **2.50** | **232.51** |
| JiT-L/16 | pixel | 459M | 200ep | 2.79 | -- |
| LDM | latent | -- | -- | 3.60 | -- |
| JiT-B/16 | pixel | 131M | 200ep | 4.37 | -- |

*Guided ImageNet 256×256 comparison with representative baselines.*

</div>

### Training Curves

<div align="center">
<img src="images/training_curves_compact.png" width="80%">
<br><br>
Training curves for the matched target ablation. Checkpoints after initialization are evaluated every 40 epochs; clean-latent variants keep lower FID and higher Inception Score than velocity counterparts.
</div>

## Key Findings

1. **Target geometry matters in latent space:** Clean-latent prediction consistently outperforms matched velocity prediction under fixed representation, architecture, and training settings.

2. **Mechanism:** Velocity prediction adds an isotropic covariance floor and amplifies low-variance latent directions, while clean prediction attenuates them.

3. **Representation independence:** The advantage is not a byproduct of using a particular patch size — it holds at both /1 and /2 VAE-grid scales.

## Citation

```bibtex
@article{fu2026jlt,
  title={{JLT}: {C}lean-{L}atent {P}rediction in {L}atent {D}iffusion {T}ransformers},
  author={Anonymous},
  journal = {arXiv preprint arXiv:2605.27102},
  year={2026}
}
```

## Acknowledgements

- Li & He. "Back to Basics: Let Denoising Generative Models Denoise." arXiv:2511.13720, 2025.
- JiT GitHub: https://github.com/LTH14/JiT
- Black Forest Labs. FLUX.2 Small Decoder. HuggingFace, 2026.