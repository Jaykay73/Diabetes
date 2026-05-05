"""Attention rollout for transformer-style models."""

from __future__ import annotations

import torch


def attention_rollout(attentions: list[torch.Tensor], discard_ratio: float = 0.0) -> torch.Tensor:
    """Fuse attention heads across blocks into a rollout map."""

    if not attentions:
        raise ValueError("attentions must not be empty")
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device).unsqueeze(0)
    for attention in attentions:
        fused = attention.mean(dim=1)
        if discard_ratio > 0:
            flat = fused.reshape(fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(1) * discard_ratio), largest=False)
            flat.scatter_(1, indices, 0)
            fused = flat.view_as(fused)
        identity = torch.eye(fused.size(-1), device=fused.device).unsqueeze(0)
        fused = (fused + identity) / 2
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        result = torch.bmm(fused, result.expand(fused.size(0), -1, -1))
    return result
