"""Step 1.4: a scene of N 3D Gaussian blobs, rendered from a camera.

Like gaussians2d.py, every blob is evaluated at every pixel, so keep images small
(64-128 px) and blob counts to a few thousand.
"""

import math
import random

import torch
from torch import nn

from gaussian_slam.geometry import Camera, covariance_3d, project, world_to_camera
from gaussian_slam.metrics import psnr

NEAR = 0.2  # blobs closer than this to the camera are dropped
BLUR = 0.3  # added to the 2D ellipse diagonal so no blob is smaller than a pixel
MAX_ALPHA = 0.99  # no blob is fully opaque, so the light behind never hits exactly 0


def render(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    cam: Camera,
    background: torch.Tensor | None = None,
):
    """Render blobs with valid parameters. Returns image (H, W, 3), depth (H, W), alpha (H, W).

    means (N, 3), quats (N, 4), scales (N, 3) > 0, colors (N, 3) in [0, 1], opacities (N,) in [0, 1].
    depth is the alpha-weighted depth; divide by alpha to get the depth of covered pixels.
    """
    h, w = cam.height, cam.width
    if background is None:
        background = torch.zeros(3, dtype=means.dtype, device=means.device)

    # 1. Blob centres into the camera frame; drop blobs that are too close or behind.
    p = world_to_camera(means, cam)  # (N, 3)
    keep = p[:, 2] > NEAR
    p, colors, opacities = p[keep], colors[keep], opacities[keep]
    cov3d = covariance_3d(quats[keep], scales[keep])  # (N, 3, 3)
    mean2d = project(p, cam)  # (N, 2) pixels
    x, y, z = p.unbind(-1)

    # 2. 3D ellipsoid -> 2D ellipse: Sigma' = J W Sigma W^T J^T.
    # J = how the pixel (u, v) moves when the camera-frame point moves, at the blob centre.
    zero = torch.zeros_like(z)
    J = torch.stack(
        [cam.fx / z, zero, -cam.fx * x / z**2, zero, cam.fy / z, -cam.fy * y / z**2], dim=-1
    ).reshape(-1, 2, 3)
    JW = J @ cam.R  # (N, 2, 3)
    cov2d = JW @ cov3d @ JW.transpose(1, 2)  # (N, 2, 2)
    a = cov2d[:, 0, 0] + BLUR
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + BLUR
    det = a * c - b * b
    inv_a, inv_b, inv_c = c / det, -b / det, a / det  # inverse of [[a, b], [b, c]]

    # 3. Sort near -> far.
    order = torch.argsort(z)
    mean2d, z, colors, opacities = mean2d[order], z[order], colors[order], opacities[order]
    inv_a, inv_b, inv_c = inv_a[order], inv_b[order], inv_c[order]

    # 4. Alpha of every blob at every pixel centre -> (N, H*W), same as the 2D weight.
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    dx = xs.reshape(1, -1).to(mean2d) - mean2d[:, 0:1]
    dy = ys.reshape(1, -1).to(mean2d) - mean2d[:, 1:2]
    power = -0.5 * (inv_a[:, None] * dx**2 + 2 * inv_b[:, None] * dx * dy + inv_c[:, None] * dy**2)
    alpha = (opacities[:, None] * torch.exp(power)).clamp(max=MAX_ALPHA)

    # 5. Front-to-back blending. T_i = light left after all blobs in front of blob i.
    left_after = torch.cumprod(1 - alpha, dim=0)  # row i: product over blobs 0..i
    T = torch.cat([torch.ones_like(alpha[:1]), left_after[:-1]], dim=0)  # blobs 0..i-1
    T_final = torch.prod(1 - alpha, dim=0)  # (H*W,) light that passes every blob
    weight = alpha * T  # (N, H*W)

    image = weight.T @ colors + T_final[:, None] * background
    depth = weight.T @ z
    return image.reshape(h, w, 3), depth.reshape(h, w), (1 - T_final).reshape(h, w)


class Gaussians3D(nn.Module):
    def __init__(self, n: int, extent: float = 1.0):
        """n random blobs inside the cube [-extent, extent]^3."""
        super().__init__()
        scale = extent / n ** (1 / 3)  # so the blobs roughly fill the cube

        # Raw (unconstrained) parameters. values() maps them to valid values.
        self.means = nn.Parameter((torch.rand(n, 3) * 2 - 1) * extent)
        self.quats = nn.Parameter(torch.randn(n, 4))  # normalized inside quat_to_rotmat
        self.log_scales = nn.Parameter(torch.full((n, 3), math.log(scale)))
        self.color_raw = nn.Parameter(torch.zeros(n, 3))
        self.opacity_raw = nn.Parameter(torch.zeros(n))

    def values(self):
        """(means, quats, scales, colors, opacities) as valid values, in render()'s order."""
        return (
            self.means,
            self.quats,
            torch.exp(self.log_scales),
            torch.sigmoid(self.color_raw),
            torch.sigmoid(self.opacity_raw),
        )

    def forward(self, cam: Camera, background: torch.Tensor | None = None):
        return render(*self.values(), cam, background)


def fit(
    model: Gaussians3D,
    cams: list[Camera],
    targets: list[torch.Tensor],
    steps: int = 3000,
    lr: float = 0.01,
    lr_min_ratio: float = 0.01,
    background: torch.Tensor | None = None,
):
    """Fit the blobs to images from several cameras. Returns [(step, mean PSNR), ...].

    Each step uses one random camera. Positions get a smaller learning rate than the
    rest, because a step of 0.01 world units is a big move for a blob.

    background (3,) must match the colour behind the objects in the targets: black
    (the default, None) for synthetic_3d.py, white for the Blender dataset.
    """
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": lr * 0.1},
            {"params": [model.quats, model.log_scales, model.color_raw, model.opacity_raw]},
        ],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * lr_min_ratio
    )
    history = []
    try:
        for step in range(1, steps + 1):
            i = random.randrange(len(cams))
            image, _, _ = model(cams[i], background)
            loss = torch.mean((image - targets[i]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    scores = [
                        psnr(model(c, background)[0], t) for c, t in zip(cams, targets, strict=True)
                    ]
                history.append((step, sum(scores) / len(scores)))
                print(f"step {step:5d}  mean PSNR {history[-1][1]:.2f} dB")
    except KeyboardInterrupt:
        # Ctrl+C stops training early; the caller still gets the model as it is now.
        print(f"\ninterrupted at step {step}, stopping early")
    return history
