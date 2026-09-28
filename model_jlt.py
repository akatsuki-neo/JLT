# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func
from util.model_util import VisionRotaryEmbeddingFast, RMSNorm


def modulate(x, shift, scale):
    if shift.ndim == x.ndim - 1:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class PatchEmbed(nn.Module):
    """Image to patch embedding: single patch_size x patch_size conv, in_chans -> embed_dim."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a Tensor of timesteps with arbitrary leading shape.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: a (..., D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float().unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        # cos/sin computed in fp32 for numerical stability; cast back to t's
        # dtype so the downstream MLP (bf16 under accelerate) matches weights.
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # flash_attn.cute does not expose a dropout_p knob; JLT defaults to 0 anyway.
        assert attn_drop == 0.0, "attn_drop is unsupported by flash_attn.cute; leave it at 0.0"
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        # Keep qkv layout (B, nH, N, hd) for RMSNorm + RoPE (per-head ops).
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        # flash_attn.cute wants (B, N, nH, hd), contiguous, fp16/bf16. Force bf16
        # here regardless of upstream dtype: rope's fp32 buffers, RMSNorm's
        # fp32 weight, or mixed_precision=no can all bubble fp32 through despite
        # the outer autocast.
        attn_dtype = torch.bfloat16
        q = q.transpose(1, 2).contiguous().to(attn_dtype)
        k = k.transpose(1, 2).contiguous().to(attn_dtype)
        v = v.transpose(1, 2).contiguous().to(attn_dtype)

        # cute.flash_attn_func always returns (out, lse); lse is None when return_lse=False.
        x, _ = flash_attn_func(q, k, v, causal=False)  # (B, N, nH, hd)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JLT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JLTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if gate_msa.ndim == x.ndim - 1:
            gate_msa = gate_msa.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# Alias for backward compatibility
JiTBlock = JLTBlock
JiT = JLT


class JLT(nn.Module):
    """
    Just Latent Transformer (JLT).
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        mask_prob=0.0,
        mask_ratio=0.0,
        loop_indices=None,
        loop_count=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.mask_prob = float(mask_prob)
        self.mask_ratio = float(mask_ratio)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        self.loop_count = int(loop_count)
        self.loop_indices = None
        if loop_indices and self.loop_count > 0:
            sorted_idx = sorted(int(x) for x in loop_indices)
            if len(set(sorted_idx)) != len(sorted_idx):
                raise ValueError(f"loop_indices must be unique, got {loop_indices}.")
            for i in range(1, len(sorted_idx)):
                if sorted_idx[i] != sorted_idx[i - 1] + 1:
                    raise ValueError(f"loop_indices must be consecutive integers, got {sorted_idx}.")
            if sorted_idx[0] < 0 or sorted_idx[-1] >= depth:
                raise ValueError(f"loop_indices must lie in [0, {depth}), got {sorted_idx}.")
            self.loop_indices = tuple(sorted_idx)

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )

        # transformer
        self.blocks = nn.ModuleList([
            JLTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, return_features=False, feature_layers=None):
        """
        x: (N, C, H, W)
        t: (N,) or (N, T)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        if t_emb.ndim == 3:
            y_emb = y_emb.unsqueeze(1)
        c = t_emb + y_emb

        # forward JLT
        x = self.x_embedder(x)
        if self.training and self.mask_prob > 0.0 and self.mask_ratio > 0.0:
            B, N, _ = x.shape
            s_mask = torch.rand(B, device=x.device) < self.mask_prob
            t_mask = torch.rand(B, N, device=x.device) < self.mask_ratio
            mask = (s_mask.unsqueeze(1) & t_mask).unsqueeze(-1)
            x = torch.where(mask, self.mask_token.to(x.dtype), x)
        target_layers = set(feature_layers or [])
        features = {} if return_features else None

        if self.loop_indices is not None and self.loop_count > 0:
            a, b = self.loop_indices[0], self.loop_indices[-1]
            schedule = (
                list(range(0, a)) + list(range(a, b + 1)) * self.loop_count + list(range(b + 1, len(self.blocks)))
            )
        else:
            schedule = list(range(len(self.blocks)))
        for layer_idx in schedule:
            x = self.blocks[layer_idx](x, c, self.feat_rope)
            if return_features and layer_idx in target_layers:
                features[layer_idx] = x

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if return_features:
            return output, features
        return output


def JLT_B_1(**kwargs):
    return JLT(depth=12, hidden_size=768, num_heads=12, patch_size=1, **kwargs)

def JLT_B_2(**kwargs):
    return JLT(depth=12, hidden_size=768, num_heads=12, patch_size=2, **kwargs)

def JLT_B_16(**kwargs):
    return JLT(depth=12, hidden_size=768, num_heads=12, patch_size=16, **kwargs)

def JLT_B_32(**kwargs):
    return JLT(depth=12, hidden_size=768, num_heads=12, patch_size=32, **kwargs)

def JLT_L_1(**kwargs):
    return JLT(depth=24, hidden_size=1024, num_heads=16, patch_size=1, **kwargs)

def JLT_L_16(**kwargs):
    return JLT(depth=24, hidden_size=1024, num_heads=16, patch_size=16, **kwargs)

def JLT_L_32(**kwargs):
    return JLT(depth=24, hidden_size=1024, num_heads=16, patch_size=32, **kwargs)

def JLT_H_1(**kwargs):
    return JLT(depth=32, hidden_size=1280, num_heads=16, patch_size=1, **kwargs)

def JLT_H_16(**kwargs):
    return JLT(depth=32, hidden_size=1280, num_heads=16, patch_size=16, **kwargs)

def JLT_H_32(**kwargs):
    return JLT(depth=32, hidden_size=1280, num_heads=16, patch_size=32, **kwargs)


# Backward compatibility aliases
JiT_B_1 = JLT_B_1
JiT_B_2 = JLT_B_2
JiT_B_16 = JLT_B_16
JiT_B_32 = JLT_B_32
JiT_L_1 = JLT_L_1
JiT_L_16 = JLT_L_16
JiT_L_32 = JLT_L_32
JiT_H_1 = JLT_H_1
JiT_H_16 = JLT_H_16
JiT_H_32 = JLT_H_32


JLT_models = {
    'JLT-B/1': JLT_B_1,
    'JLT-B/2': JLT_B_2,
    'JLT-B/16': JLT_B_16,
    'JLT-B/32': JLT_B_32,
    'JLT-L/1': JLT_L_1,
    'JLT-L/16': JLT_L_16,
    'JLT-L/32': JLT_L_32,
    'JLT-H/1': JLT_H_1,
    'JLT-H/16': JLT_H_16,
    'JLT-H/32': JLT_H_32,
    # Backward compatibility
    'JiT-B/1': JLT_B_1,
    'JiT-B/2': JLT_B_2,
    'JiT-B/16': JLT_B_16,
    'JiT-B/32': JLT_B_32,
    'JiT-L/1': JLT_L_1,
    'JiT-L/16': JLT_L_16,
    'JiT-L/32': JLT_L_32,
    'JiT-H/1': JLT_H_1,
    'JiT-H/16': JLT_H_16,
    'JiT-H/32': JLT_H_32,
}

JiT_models = JLT_models
