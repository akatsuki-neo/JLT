import argparse
import math
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from accelerate import Accelerator, InitProcessGroupKwargs

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from latent_dataset import Flux2LatentDataset


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--mask_prob', type=float, default=0.0,
                        help='Per-sample probability of MAE-style token masking (training only). '
                             '0 disables. Replaces post-embedding visual tokens with a learnable mask_token.')
    parser.add_argument('--mask_ratio', type=float, default=0.0,
                        help='Per-token Bernoulli probability of being replaced with mask_token, within selected samples. '
                             'Only used when --mask_prob > 0. Inference/sampling never masks.')
    parser.add_argument('--loop_indices', type=str, default='',
                        help='Comma-separated consecutive non-negative integers marking the loop body. '
                             'These are layer indices. E.g. "1,2" loops layers 1 and 2 together.')
    parser.add_argument('--loop_count', type=int, default=0,
                        help='Total number of iterations of the loop body in place of one normal pass. '
                             '0 disables looping; N>=1 unrolls the body N times (N=1 == default order).')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Micro batch per GPU (effective = batch_size * # GPUs * accum_iter)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation steps (DeepSpeed ZeRO-aware).')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max grad norm for gradient clipping (0 disables). Under DeepSpeed, '
                             'accelerate cannot clip here — also set `gradient_clipping` in the DS config.')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--async_timesteps', action='store_true',
                        help='Use tokenwise timesteps during training noise injection while sharing one noise sample per image.')
    parser.add_argument('--async_timestep_drop', default=0.0, type=float,
                        help='Per-sample probability of falling back from tokenwise timesteps to a single shared timestep.')
    parser.add_argument('--ema_feat_align_weight', default=0.0, type=float,
                        help='Weight for EMA teacher feature alignment loss; disabled when <= 0.')
    parser.add_argument('--ema_feat_align_teacher_layers', default='', type=str,
                        help='Comma-separated EMA teacher block indices used for feature alignment.')
    parser.add_argument('--ema_feat_align_student_layers', default='', type=str,
                        help='Comma-separated student block indices used for feature alignment.')
    parser.add_argument('--vae_type', default='identity', type=str, choices=['identity', 'flux2'],
                        help='VAE wrapper: "identity" = pixel-space pass-through, "flux2" = FLUX.2 VAE (128-ch, 16x spatial compression).')
    parser.add_argument('--vae_model_name_or_path',
                        default='black-forest-labs/FLUX.2-klein-4B', type=str,
                        help='HF repo id or local path for the VAE (ignored when --vae_type identity).')
    parser.add_argument('--vae_subfolder', default='vae', type=str,
                        help='Subfolder inside the VAE checkpoint that contains the weights.')
    parser.add_argument('--t_eps', default=5e-2, type=float,
                        help='Denominator floor for x-prediction velocity reconstruction. Ignored under --flow_matching.')
    parser.add_argument('--flow_matching', action='store_true',
                        help='Train with direct v-prediction (flow matching): target v = x - e, no 1/(1-t) factor.')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset (ImageFolder root; Parquet dir; or pre-encoded latent shard dir)')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--use_latent_cache', action='store_true',
                        help='Load pre-encoded latent shards (safetensors) produced by encode_vae_latents.py. '
                             'Dataset returns (latent, label); combine with identity-VAE denoiser.')
    parser.add_argument('--use_parquet', action='store_true',
                        help='Load ImageNet from HuggingFace-style parquet files (memory-mapped via `datasets`).')
    parser.add_argument('--cache_dir', default='./hf_cache', type=str,
                        help='HF datasets cache dir (used by --use_parquet).')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)

    # wandb
    parser.add_argument('--wandb_project', default='JiT', type=str,
                        help='wandb project name')
    parser.add_argument('--wandb_name', default=None, type=str,
                        help='wandb run name; defaults to output_dir basename')
    parser.add_argument('--wandb_mode', default='online', type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='wandb mode: online/offline/disabled')

    # distributed
    parser.add_argument('--collective_timeout_hours', default=2.0, type=float,
                        help='NCCL/gloo collective timeout. Must exceed the worst-case FID eval '
                             'time on rank 0 (torch_fidelity over --num_images).')

    return parser


def main(args):
    # Extend collective timeout: during evaluate(), non-main ranks idle at
    # wait_for_everyone while rank 0 runs torch_fidelity over 50k images; the
    # default 10-min NCCL watchdog can trip and abort the whole group.
    pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=args.collective_timeout_hours))
    accelerator = Accelerator(
        log_with='wandb',
        kwargs_handlers=[pg_kwargs],
        gradient_accumulation_steps=args.accum_iter,
    )
    # DS plugin maintains its own gradient_accumulation_steps. Accelerate usually
    # syncs it, but force-align it here so DeepSpeedEngine sees the exact value.
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = args.accum_iter
    misc.setup_for_distributed(accelerator.is_main_process)

    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = accelerator.device

    # Set seeds for reproducibility
    seed = args.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Set up logging via accelerate's tracker (wandb backend; main process only).
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={
            'wandb': {
                'name': args.wandb_name or os.path.basename(args.output_dir.rstrip('/')),
                'dir': args.output_dir,
                'mode': args.wandb_mode,
            }
        },
    )

    # Image transform outputs float32 in [-1, 1]; Flux2LatentDataset bypasses this
    # path and yields pre-encoded latents directly.
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.use_latent_cache:
        dataset_train = Flux2LatentDataset(args.data_path, use_flip=True)
    elif args.use_parquet:
        # Lazy-import so non-parquet users don't need pyarrow / datasets installed.
        from parquet_dataset import HuggingFaceImageNetDataset
        os.makedirs(args.cache_dir, exist_ok=True)
        dataset_train = HuggingFaceImageNetDataset(
            data_dir=args.data_path,
            split='train',
            transform=transform_train,
            cache_dir=args.cache_dir,
        )
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_mem,
        'drop_last': True,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True
    data_loader_train = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    eff_batch_size = args.batch_size * accelerator.num_processes * args.accum_iter
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # One-shot prepare(model, optimizer, data_loader). Under DeepSpeed ZeRO-2
    # this is the only sequence that satisfies both:
    #   * "zero stage 2 requires an optimizer"  (need optimizer, not DummyOptim)
    #   * "train_micro_batch_size_per_gpu"      (need a loader to infer bs)
    model, optimizer, data_loader_train = accelerator.prepare(model, optimizer, data_loader_train)
    model_without_ddp = accelerator.unwrap_model(model)

    updates_per_epoch = math.ceil(len(data_loader_train) / args.accum_iter)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        # weights_only=False: needed on torch>=2.6 to restore argparse.Namespace,
        # EMA tensor lists, and optimizer state (non-tensor pickled objects).
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        # EMA tensors kept in fp32 so decay=0.9999 (1-decay=1e-4) is representable.
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device=device, dtype=torch.float32)
                                          for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device=device, dtype=torch.float32)
                                          for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        # EMA tensors kept in fp32 so decay=0.9999 (1-decay=1e-4) is representable.
        model_without_ddp.ema_params1 = [p.detach().float().clone() for p in model_without_ddp.parameters()]
        model_without_ddp.ema_params2 = [p.detach().float().clone() for p in model_without_ddp.parameters()]
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(
                    accelerator,
                    model_without_ddp,
                    args,
                    args.start_epoch,
                    batch_size=args.gen_bsz,
                    log_step=args.start_epoch * updates_per_epoch,
                )
        accelerator.end_training()
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(data_loader_train, 'set_epoch'):
            data_loader_train.set_epoch(epoch)

        train_one_epoch(accelerator, model, model_without_ddp, data_loader_train, optimizer, epoch, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(
                    accelerator,
                    model_without_ddp,
                    args,
                    epoch,
                    batch_size=args.gen_bsz,
                    log_step=(epoch + 1) * updates_per_epoch,
                )
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

    accelerator.end_training()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
