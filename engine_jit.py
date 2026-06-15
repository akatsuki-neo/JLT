import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def train_one_epoch(accelerator, model, model_without_ddp, data_loader, optimizer, epoch, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    updates_per_epoch = math.ceil(len(data_loader) / args.accum_iter)
    optimizer_step = epoch * updates_per_epoch

    optimizer.zero_grad()

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Gradient-accumulation context; under DeepSpeed this also delegates to
        # the engine's accumulation boundary logic. EMA / logging below key off
        # accelerator.sync_gradients so they only fire on real optimizer steps.
        with accelerator.accumulate(model):
            # Dataset-side transform already yields float32 in [-1, 1] for image
            # paths; latent shards yield pre-encoded floats — pass through.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            accelerator.backward(loss)
            if accelerator.sync_gradients and getattr(args, 'grad_clip', 0.0) > 0.0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            optimizer_step += 1
            torch.cuda.synchronize()
            model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if accelerator.sync_gradients and optimizer_step % args.log_freq == 0:
            accelerator.log({'train_loss': loss_value_reduce, 'lr': lr}, step=optimizer_step)


def evaluate(accelerator, model_without_ddp, args, epoch, batch_size=64, log_step=None):

    model_without_ddp.eval()
    world_size = accelerator.num_processes
    local_rank = accelerator.process_index
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if accelerator.is_main_process and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    accelerator.wait_for_everyone()

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        # EMA is fp32; cast to the training weight dtype before swap-in.
        ema_state_dict[name] = model_without_ddp.ema_params1[i].to(ema_state_dict[name].dtype)
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().to(accelerator.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        accelerator.wait_for_everyone()

        # denormalize images; cast to fp32 because numpy can't consume bf16
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().float().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    accelerator.wait_for_everyone()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS (main process only; torch_fidelity reads the full folder)
    if accelerator.is_main_process:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        if log_step is None:
            log_step = epoch + 1
        accelerator.log(
            {f'fid{postfix}': fid, f'is{postfix}': inception_score},
            step=log_step,
        )
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    accelerator.wait_for_everyone()
