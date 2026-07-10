"""Tiny transformer decoder used for the grokking replication."""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("attn_mask", mask, persistent=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        mask = self.attn_mask[:t, :t]
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_f(h)
        # Loss is applied at the final position only (the equals slot).
        return self.head(h[:, -1, :])
