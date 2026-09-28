"""Efficient random-access dataset for pre-encoded FLUX2 latents.

Latents are produced by ``prepare_ref.py`` and stored as sharded
``safetensors`` files. Each shard contains:

  - ``latents``      : (N, C, H, W) fp16, BN-normalized FLUX2 patchified latents
  - ``latents_flip`` : (N, C, H, W) fp16, same but for the horizontally flipped image
  - ``labels``       : (N,) int32 class indices

Layout on disk::

    <root>/
      latents_rank00_shard000.safetensors
      latents_rank00_shard001.safetensors
      ...
      latents_rank07_shard017.safetensors
      shard_index.json   # cached global index (built lazily on first run)

Note: FLUX2 VAE has a built-in BatchNorm (``vae.bn``) that normalizes the
encoded latents to ~zero-mean / ~unit-variance at encode time. The encoder
mirrors that BN normalization, so the latents
on disk are ALREADY normalized — there is no separate stats / normalize step
inside this dataset.

Random-access design notes
--------------------------

* ``safetensors.safe_open`` returns a memory-mapped handle. Slice reads are
  O(1) (`mmap` + memcpy), so per-sample access is very cheap *after* the file
  has been opened. Opening itself parses the JSON header — not free.
* We therefore cache an open handle per worker process. ``DataLoader`` workers
  are forked, so each worker has its own dict and there is no inter-worker
  contention. The cache is unbounded; safetensors handles are tiny (~one
  ``mmap`` per file), and the total number of shards is small (a few hundred).
* Index construction is the only O(num_shards) work at start-up. We persist
  the resulting list of ``(shard_path, shard_size)`` tuples to
  ``shard_index.json`` so subsequent runs skip header reads entirely.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import Dataset


_INDEX_FILENAME = "shard_index.json"


# ---------------------------------------------------------------------------
# Per-worker handle cache
# ---------------------------------------------------------------------------

# DataLoader workers are forked, so this dict is naturally per-worker.
_SAFE_HANDLES: Dict[str, "safe_open"] = {}


def _get_handle(path: str):
    """Return a cached, memory-mapped safetensors handle for ``path``."""
    handle = _SAFE_HANDLES.get(path)
    if handle is None:
        handle = safe_open(path, framework="pt", device="cpu")
        _SAFE_HANDLES[path] = handle
    return handle


def _close_all_handles() -> None:
    """Optional helper for tests / shutdown."""
    _SAFE_HANDLES.clear()


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _read_shard_size(path: str) -> int:
    """Return number of samples stored in a shard via its labels tensor."""
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_slice("labels").get_shape()[0]


def build_shard_index(data_dir: str, force: bool = False) -> List[Tuple[str, int]]:
    """Scan ``data_dir`` for shards and return ``[(path, num_samples), ...]``.

    Caches the result to ``shard_index.json`` so subsequent loads are O(1).
    """
    cache_path = os.path.join(data_dir, _INDEX_FILENAME)
    shard_paths = sorted(glob(os.path.join(data_dir, "latents_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No latent shards found under {data_dir}")

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_paths = [os.path.join(data_dir, name) for name, _ in cached]
            if cached_paths == shard_paths:
                return [(p, int(n)) for p, (_, n) in zip(shard_paths, cached)]
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # rebuild

    index: List[Tuple[str, int]] = []
    for path in shard_paths:
        index.append((path, _read_shard_size(path)))

    try:
        with open(cache_path, "w") as f:
            json.dump([(os.path.basename(p), n) for p, n in index], f)
    except OSError:
        pass  # not fatal — read-only filesystem etc.
    return index


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Flux2LatentDataset(Dataset):
    """Random-access dataset over pre-encoded FLUX2 latents.

    Parameters
    ----------
    data_dir:
        Directory produced by ``prepare_ref.py`` (contains
        ``latents_rank*_shard*.safetensors``).
    use_flip:
        If ``True`` (default), randomly return the cached horizontally-flipped
        latent half the time — equivalent to ``RandomHorizontalFlip(p=0.5)``
        on raw pixels but free at training time.
    """

    def __init__(self, data_dir: str, use_flip: bool = True) -> None:
        self.data_dir = data_dir
        self.use_flip = use_flip

        self._shards: List[Tuple[str, int]] = build_shard_index(data_dir)
        self._cum_sizes = np.zeros(len(self._shards) + 1, dtype=np.int64)
        for i, (_, n) in enumerate(self._shards):
            self._cum_sizes[i + 1] = self._cum_sizes[i] + n
        self._length = int(self._cum_sizes[-1])

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> Tuple[str, int]:
        """Map a global index to ``(shard_path, local_idx)``."""
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        shard_idx = int(np.searchsorted(self._cum_sizes, idx, side="right") - 1)
        local_idx = idx - int(self._cum_sizes[shard_idx])
        return self._shards[shard_idx][0], local_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_path, local_idx = self._locate(idx)
        handle = _get_handle(shard_path)
        key = "latents_flip" if (self.use_flip and torch.rand(()).item() > 0.5) else "latents"
        latent = handle.get_slice(key)[local_idx : local_idx + 1].squeeze(0).to(torch.float32)
        label = handle.get_slice("labels")[local_idx : local_idx + 1].squeeze(0).to(torch.long)
        return latent, label


__all__ = ["Flux2LatentDataset", "build_shard_index"]
