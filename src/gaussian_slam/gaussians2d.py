"""Step 1.2: an image made of N 2D Gaussian blobs, fitted to a photo."""

import math

import torch
from torch import nn

from gaussian_slam.metrics import psnr


class Gaussians2D(nn.Module):
    def __init__(self, n: int, height: int, width: int):
        super().__init__()
        self.height, self.width = height, width
        sigma = 0.5 * math.sqrt(height * width / n)  # so the blobs roughly cover the image

        # Raw (unconstrained) parameters. forward() maps them to valid values.
        self.mean_raw = nn.Parameter(torch.logit(torch.rand(n, 2).clamp(0.01, 0.99)))
        self.log_scale = nn.Parameter(torch.full((n, 2), math.log(sigma)))
        self.angle = nn.Parameter(torch.rand(n) * math.pi)
        self.color_raw = nn.Parameter(torch.zeros(n, 3))
        self.opacity_raw = nn.Parameter(torch.zeros(n))

    def forward(self) -> torch.Tensor:
        h, w = self.height, self.width

        # 1. Raw -> valid values.
        size = torch.tensor([w, h], device=self.mean_raw.device)
        mean = torch.sigmoid(self.mean_raw) * size  # (N, 2) pixels, inside the image
        inv_var = torch.exp(-2 * self.log_scale)  # (N, 2) = 1 / s^2, always > 0
        color = torch.sigmoid(self.color_raw)  # (N, 3) in [0, 1]
        opacity = torch.sigmoid(self.opacity_raw)  # (N,)  in [0, 1]

        # 2. Inverse covariance [[a, b], [b, c]] = R diag(1/sx^2, 1/sy^2) R^T.
        cos, sin = torch.cos(self.angle), torch.sin(self.angle)
        a = cos**2 * inv_var[:, 0] + sin**2 * inv_var[:, 1]
        b = cos * sin * (inv_var[:, 0] - inv_var[:, 1])
        c = sin**2 * inv_var[:, 0] + cos**2 * inv_var[:, 1]

        # 3. Offset of every pixel centre from every blob centre -> (N, H*W).
        ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
        dx = xs.reshape(1, -1).to(mean) - mean[:, 0:1]
        dy = ys.reshape(1, -1).to(mean) - mean[:, 1:2]

        # 4. Weight of each blob at each pixel, then add up the colours.
        power = -0.5 * (a[:, None] * dx**2 + 2 * b[:, None] * dx * dy + c[:, None] * dy**2)
        weight = opacity[:, None] * torch.exp(power)  # (N, H*W)
        return (weight.T @ color).reshape(h, w, 3)  # (H*W, N) @ (N, 3)


def fit(model: Gaussians2D, target: torch.Tensor, steps: int = 2000, lr: float = 0.01):
    """Gradient descent on the mean squared error. Returns [(step, PSNR), ...]."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        pred = model()
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append((step, psnr(pred.detach(), target)))
            print(f"step {step:5d}  PSNR {history[-1][1]:.2f} dB")
    return history
