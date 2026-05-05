"""Gradient-based image counterfactual explanations."""

from __future__ import annotations

import torch


def generate_gradient_counterfactual(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_value: float,
    steps: int = 80,
    lr: float = 1e-2,
    l1_weight: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find a small image delta that moves a scalar model output toward target_value."""

    model.eval()
    device = next(model.parameters()).device
    base = image_tensor.detach().clone().unsqueeze(0).to(device)
    delta = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    target = torch.tensor(float(target_value), device=device)
    for _ in range(steps):
        candidate = (base + delta).clamp(-3.0, 3.0)
        output = model(candidate).reshape(-1)[0]
        loss = (output - target).pow(2) + l1_weight * delta.abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    counterfactual = (base + delta).detach().squeeze(0).cpu()
    difference = delta.detach().abs().squeeze(0).mean(dim=0).cpu()
    return counterfactual, difference
