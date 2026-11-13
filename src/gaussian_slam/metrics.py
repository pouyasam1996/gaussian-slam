import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]. Higher is better."""
    mse = torch.mean((pred.clamp(0, 1) - target) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5):
    """Structural similarity of two (H, W, 3) images in [0, 1], as a differentiable tensor.

    1 = identical. Compares local mean, contrast and structure in an 11x11 Gaussian window,
    like the loss of the original 3DGS.
    """
    c1, c2 = 0.01**2, 0.03**2
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()

    def blur(t):  # per-channel Gaussian average
        return F.conv2d(t, kernel, padding=window // 2, groups=3)

    x, y = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]  # (1, 3, H, W)
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return s.mean()
