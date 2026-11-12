import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]. Higher is better."""
    mse = torch.mean((pred.clamp(0, 1) - target) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)
