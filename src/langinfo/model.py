"""Causal decoder-only Transformer architecture (MonoT) and sampling utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class Block(nn.Module):
    """Pre-norm causal Transformer block with GELU MLP."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head}).")
        self.n_head = n_head
        self.dropout = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.out = nn.Linear(4 * d_model, d_model, bias=False)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(dim, dim=2)
        head_dim = dim // self.n_head
        q = q.view(bsz, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, head_dim).transpose(1, 2)

        attn_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_p, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        x = x + self.resid_drop(self.proj(y))
        x = x + self.mlp_drop(self.out(F.gelu(self.fc(self.ln2(x)))))
        return x


class MonoT(nn.Module):
    """Monolingual causal Transformer language model with tied input/output embeddings.

    Parameters
    ----------
    vocab_size : int
        Size of the subword vocabulary.
    d_model : int
        Embedding and hidden dimension.
    n_layer : int
        Number of causal Transformer blocks.
    n_head : int
        Number of attention heads per block.
    block_size : int
        Maximum context window length in tokens.
    dropout : float, default=0.1
        Dropout rate applied to embeddings, attention, and residual branches.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.wte.weight

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token index sequences."""
        seq_len = idx.shape[1]
        if seq_len > self.block:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block}.")
        pos = torch.arange(seq_len, device=idx.device)
        x = self.emb_drop(self.wte(idx) + self.wpe(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.head(self.lnf(x))


@torch.no_grad()
def generate_tokens(
    model: MonoT,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    eos_id: int = EOS_ID,
    forbidden_ids: Iterable[int] = (PAD_ID, UNK_ID, BOS_ID),
    seed: int | None = None,
) -> list[int]:
    """Autoregressively sample continuation token IDs from a trained MonoT model."""
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    forbidden = list(forbidden_ids)

    for _ in range(max_new_tokens):
        logits = model(x[:, -model.block :])[:, -1, :].float()
        if forbidden:
            logits[:, forbidden] = -float("inf")

        if repetition_penalty != 1.0:
            for tid in set(x[0, -64:].tolist()):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature <= 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1, generator=gen)
            next_id = sorted_idx.gather(-1, sampled)

        if next_id.item() == eos_id:
            break
        x = torch.cat([x, next_id], dim=1)

    return x[0, len(prompt_ids) :].tolist()
