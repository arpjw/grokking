"""Generate all (a, b) pairs mod p and split into train/test by fraction."""
from __future__ import annotations

import torch


def make_data(p: int, frac_train: float, seed: int):
    """Return train_x, train_y, test_x, test_y tensors.

    Each example is [a, b, eq_token] of length 3, target is (a + b) mod p.
    Vocab is 0..p-1 for numbers plus a single equals token with id = p, so
    the model's vocab size is p + 1.
    """
    eq_token = p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    y = (a + b) % p
    eq = torch.full_like(a, eq_token)
    x = torch.stack([a, b, eq], dim=1)

    n = x.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(round(frac_train * n))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def vocab_size(p: int) -> int:
    return p + 1
