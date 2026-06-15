import torch
import torch.nn as nn
from torch.func import functional_call
from model_jit import JiT_models
from vae import build_vae


MODEL_REGISTRY = {}
MODEL_REGISTRY.update(JiT_models)


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        vae_type = args.vae_type
        if vae_type == 'flux2':
            self.vae = build_vae(
                'flux2',
                model_name_or_path=getattr(args, 'vae_model_name_or_path', 'black-forest-labs/FLUX.2-klein-4B'),
                subfolder=getattr(args, 'vae_subfolder', 'vae'),
            )
        else:
            self.vae = build_vae(vae_type, in_channels=3)
        self.vae.requires_grad_(False)
        self.vae.eval()

        latent_size = args.img_size // self.vae.spatial_compression

        model_kwargs = dict(
            input_size=latent_size,
            in_channels=self.vae.latent_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        mask_prob = float(getattr(args, 'mask_prob', 0.0) or 0.0)
        if mask_prob > 0.0:
            model_kwargs['mask_prob'] = mask_prob
            model_kwargs['mask_ratio'] = float(getattr(args, 'mask_ratio', 0.0) or 0.0)
        loop_count = int(getattr(args, 'loop_count', 0) or 0)
        if loop_count > 0:
            loop_indices_raw = str(getattr(args, 'loop_indices', '') or '')
            loop_indices = [int(x) for x in loop_indices_raw.split(',') if x.strip()]
            if loop_indices:
                model_kwargs['loop_indices'] = loop_indices
                model_kwargs['loop_count'] = loop_count
        self.net = MODEL_REGISTRY[args.model](**model_kwargs)
        self.img_size = args.img_size
        self.latent_size = latent_size
        self.latent_channels = self.vae.latent_channels
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        # Flow matching: network outputs velocity v = x - e directly (t:0->1 = noise->clean).
        # x-prediction path reconstructs v via (x_pred - z) / (1 - t) and needs t_eps floor;
        # flow matching skips that division and ignores t_eps.
        self.flow_matching = bool(getattr(args, 'flow_matching', False))
        self.async_timesteps = bool(getattr(args, 'async_timesteps', False))
        self.async_timestep_drop = float(getattr(args, 'async_timestep_drop', 0.0))
        self.ema_feat_align_weight = float(getattr(args, 'ema_feat_align_weight', 0.0))
        self.ema_feat_align_teacher_layers = self._parse_layer_list(getattr(args, 'ema_feat_align_teacher_layers', ''))
        self.ema_feat_align_student_layers = self._parse_layer_list(getattr(args, 'ema_feat_align_student_layers', ''))
        if len(self.ema_feat_align_teacher_layers) != len(self.ema_feat_align_student_layers):
            raise ValueError("ema_feat_align_teacher_layers and ema_feat_align_student_layers must have the same length.")

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.use_latent_cache = args.use_latent_cache

    @staticmethod
    def _parse_layer_list(spec) -> list[int]:
        if spec is None or spec == '':
            return []
        if isinstance(spec, (list, tuple)):
            return [int(v) for v in spec]
        return [int(v.strip()) for v in str(spec).split(',') if v.strip()]

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _token_patch_size(self) -> int:
        if hasattr(self.net, 'patch_size'):
            return int(self.net.patch_size)
        if hasattr(self.net, 'latent_patch_size'):
            return int(self.net.latent_patch_size)
        return 1

    def _sample_tokenwise_t(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_size = self._token_patch_size()
        _, _, height, width = x.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Input spatial size {(height, width)} is not divisible by token patch size {patch_size}.")
        token_h = height // patch_size
        token_w = width // patch_size
        token_t = self.sample_t(x.size(0) * token_h * token_w, device=x.device).to(dtype=x.dtype).view(x.size(0), token_h * token_w)
        if self.async_timestep_drop > 0.0:
            disable_async = torch.rand(x.size(0), device=x.device) < self.async_timestep_drop
            if disable_async.any():
                fallback_t = self.sample_t(int(disable_async.sum().item()), device=x.device).to(dtype=x.dtype)
                token_t[disable_async] = fallback_t.unsqueeze(-1)
        t_map = token_t.view(x.size(0), 1, token_h, token_w)
        if patch_size > 1:
            t_map = t_map.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
        return token_t, t_map

    def _t_tokens_to_map(self, x: torch.Tensor, t_tokens: torch.Tensor) -> torch.Tensor:
        if t_tokens.ndim == 1:
            return t_tokens.to(dtype=x.dtype).view(-1, *([1] * (x.ndim - 1)))
        patch_size = self._token_patch_size()
        _, _, height, width = x.shape
        token_h = height // patch_size
        token_w = width // patch_size
        t_map = t_tokens.to(dtype=x.dtype).view(x.size(0), 1, token_h, token_w)
        if patch_size > 1:
            t_map = t_map.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
        return t_map

    def _use_ema_feat_align(self) -> bool:
        return (
            self.training
            and self.ema_feat_align_weight > 0.0
            and len(self.ema_feat_align_teacher_layers) > 0
            and self.ema_params1 is not None
        )

    def _build_ema_teacher_t(self, x: torch.Tensor, student_t_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_sample_t = self.sample_t(x.size(0), device=x.device).to(dtype=x.dtype)
        if student_t_tokens.ndim == 1:
            teacher_t_tokens = torch.minimum(teacher_sample_t, student_t_tokens.to(dtype=x.dtype))
        else:
            teacher_t_tokens = torch.minimum(teacher_sample_t.unsqueeze(1), student_t_tokens.to(dtype=x.dtype))
        teacher_t = self._t_tokens_to_map(x, teacher_t_tokens)
        return teacher_t_tokens, teacher_t

    def _ema_net_named_parameters(self) -> dict[str, torch.Tensor]:
        ema_named_params = {}
        for (name, param), ema_param in zip(self.named_parameters(), self.ema_params1):
            if name.startswith('net.'):
                ema_named_params[name[4:]] = ema_param.detach().to(device=param.device, dtype=param.dtype)
        return ema_named_params

    def _ema_feature_alignment_loss(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        labels: torch.Tensor,
        student_t_tokens: torch.Tensor,
        student_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        teacher_t_tokens, teacher_t = self._build_ema_teacher_t(x, student_t_tokens)
        teacher_z = teacher_t * x + (1 - teacher_t) * e
        ema_params = self._ema_net_named_parameters()
        ema_buffers = dict(self.net.named_buffers())
        with torch.no_grad():
            _, teacher_features = functional_call(
                self.net,
                (ema_params, ema_buffers),
                (teacher_z, teacher_t_tokens, labels),
                {'return_features': True, 'feature_layers': self.ema_feat_align_teacher_layers},
            )

        feat_loss = x.new_zeros(())
        for teacher_layer, student_layer in zip(self.ema_feat_align_teacher_layers, self.ema_feat_align_student_layers):
            teacher_feat = teacher_features[teacher_layer]
            student_feat = student_features[student_layer]
            cosine = torch.nn.functional.cosine_similarity(
                student_feat.float(),
                teacher_feat.float(),
                dim=-1,
                eps=1e-8,
            )
            feat_loss = feat_loss + (1.0 - cosine).mean()
        return feat_loss / len(self.ema_feat_align_teacher_layers)

    def forward(self, x, labels):
        with torch.no_grad():
            if not self.use_latent_cache:
                x = self.vae.encode(x)
        model_dtype = next(self.net.parameters()).dtype
        x = x.to(dtype=model_dtype)

        labels_dropped = self.drop_labels(labels) if self.training else labels

        if self.async_timesteps:
            t_tokens, t = self._sample_tokenwise_t(x)
        else:
            t = self.sample_t(x.size(0), device=x.device).to(dtype=x.dtype).view(-1, *([1] * (x.ndim - 1)))
            t_tokens = t.flatten()
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        # Target velocity. In both parameterizations v = dz/dt = x - e; the xpred
        # form (x - z)/(1 - t) is algebraically equivalent but needs t_eps to
        # avoid 0/0 as t -> 1. Flow matching uses the direct form.
        if self.flow_matching:
            v = x - e
        else:
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

        use_ema_feat_align = self._use_ema_feat_align()
        if use_ema_feat_align:
            net_out, student_features = self.net(
                z,
                t_tokens,
                labels_dropped,
                return_features=True,
                feature_layers=self.ema_feat_align_student_layers,
            )
        else:
            net_out = self.net(z, t_tokens, labels_dropped)
        # Interpret network output: flow matching -> direct v; xpred -> reconstruct v from x_pred.
        if self.flow_matching:
            v_pred = net_out
        else:
            v_pred = (net_out - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        if use_ema_feat_align:
            feat_loss = self._ema_feature_alignment_loss(x, e, labels_dropped, t_tokens, student_features)
            loss = loss + self.ema_feat_align_weight * feat_loss

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, self.latent_channels, self.latent_size, self.latent_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return self.vae.decode(z)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        out_cond = self.net(z, t.flatten(), labels)
        # unconditional
        out_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))

        if self.flow_matching:
            v_cond, v_uncond = out_cond, out_uncond
        else:
            v_cond = (out_cond - z) / (1.0 - t).clamp_min(self.t_eps)
            v_uncond = (out_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        # EMA kept in fp32: with bf16/fp16 training weights, decay=0.9999 would
        # have 1-decay=1e-4 unrepresentable in bf16, making the EMA drift/freeze.
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src.detach().float(), alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src.detach().float(), alpha=1 - self.ema_decay2)
