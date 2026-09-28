"""VAE wrappers for JLT.

Contract (all subclasses):
  encode(image) -> latent ready to feed the diffusion net
  decode(latent_from_net) -> image
  latent_channels: int
  spatial_compression: int       # how much smaller the latent is per side
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Identity (pixel-space pass-through)
# ---------------------------------------------------------------------------

class IdentityVAE(nn.Module):
    """Pixel-space wrapped as a trivial VAE: encode/decode are identity."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.latent_channels = in_channels
        self.spatial_compression = 1

    def encode(self, x):
        return x

    def decode(self, z):
        return z


# ---------------------------------------------------------------------------
# FLUX.2 VAE (128-channel, 16x spatial downsample, BN-normalized output)
# ---------------------------------------------------------------------------

def _flux2_retrieve_latents(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _flux2_patchify(latents):
    """pixel-shuffle 2x2: (B, C, H, W) -> (B, 4C, H/2, W/2)."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def _flux2_unpatchify(latents):
    """inverse of _flux2_patchify."""
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, C // 4, H * 2, W * 2)


class Flux2VAE(nn.Module):
    """Black-Forest FLUX.2 VAE wrapped as a JLT VAE layer.

    Encoding (image -> latent ready for DiT):
      1. vae.encode(x).latent_dist.mode()                       # deterministic
      2. 2x2 pixel-shuffle: (B, C, H, W) -> (B, 4C, H/2, W/2)
      3. BN-normalize with the VAE's own `vae.bn` running stats:
             (latent - bn.running_mean) / sqrt(bn.running_var + eps)

    Decoding reverses all three steps.

    Final latent: 128 channels, spatial compression = encoder_down * 2
    (encoder_down = 2^(len(block_out_channels)-1); FLUX.2 klein -> 16x total).

    VAE weights bypass nn.Module registration (via object.__setattr__), so they
    do NOT land in state_dict / EMA / optimizer / DDP reducer buckets. The
    `_apply` override ensures `.to(device)` / `.cuda()` still reach the VAE.
    """

    def __init__(
        self,
        model_name_or_path: str = 'black-forest-labs/FLUX.2-klein-4B',
        subfolder: str = 'vae',
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKLFlux2
        except ImportError as exc:
            raise ImportError(
                "diffusers with AutoencoderKLFlux2 is required for Flux2VAE"
            ) from exc

        load_kwargs = {'subfolder': subfolder} if subfolder else {}
        vae = AutoencoderKLFlux2.from_pretrained(model_name_or_path, **load_kwargs)
        vae.requires_grad_(False)
        vae.eval()

        # Bypass Module registration: _vae is a plain attribute, not a submodule.
        object.__setattr__(self, '_vae', vae)

        # Latent spec exposed to the denoiser/net.
        self.latent_channels = 128
        encoder_down = 2 ** (len(vae.config.block_out_channels) - 1)
        self.spatial_compression = encoder_down * 2  # patchify halves again

    def _apply(self, fn):
        """Route device/dtype migrations (cuda(), to(device), bfloat16(), …) to _vae too."""
        super()._apply(fn)
        self._vae._apply(fn)
        return self

    def _bn_stats(self, ref):
        """Return (mean, std) shaped (1, C, 1, 1), matched to ref.device/dtype.

        std uses +eps to match the offline encoder (encode_vae_latents.py), so
        online decode mirrors the exact BN normalization applied at cache time.
        """
        vae = self._vae
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
        std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1).to(ref.device, ref.dtype)
            + vae.config.batch_norm_eps
        )
        return mean, std

    @torch.no_grad()
    def encode(self, x):
        vae = self._vae
        vae_param = next(vae.parameters())
        x = x.to(device=vae_param.device, dtype=vae_param.dtype)
        latents = _flux2_retrieve_latents(vae.encode(x))
        latents = _flux2_patchify(latents)
        mean, std = self._bn_stats(latents)
        return (latents - mean) / std

    @torch.no_grad()
    def decode(self, z):
        vae = self._vae
        vae_param = next(vae.parameters())
        z = z.to(device=vae_param.device, dtype=vae_param.dtype)
        mean, std = self._bn_stats(z)
        latents = z * std + mean
        latents = _flux2_unpatchify(latents)
        # diffusers VAE decode: return_dict=False -> (sample,)
        return vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VAE_registry = {
    "identity": IdentityVAE,
    "flux2": Flux2VAE,
}


def build_vae(name: str, **kwargs) -> nn.Module:
    if name not in VAE_registry:
        raise ValueError(f"Unknown VAE '{name}'. Available: {sorted(VAE_registry)}")
    return VAE_registry[name](**kwargs)
