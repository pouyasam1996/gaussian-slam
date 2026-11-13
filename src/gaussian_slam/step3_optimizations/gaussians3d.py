"""Step 3: step 2's blobs, made to work on a real scene.

What changes from step 2:
  renderer     tiles.render_tiled(): the same picture with far less memory (see tiles.py)
  colour       view-dependent, through spherical harmonics (SH): shiny or shaded surfaces
               look different from different sides
  loss         0.8 L1 + 0.2 (1 - SSIM), the loss of the original 3DGS: SSIM rewards sharp
               edges and local contrast, which a per-pixel error alone blurs
  learning     one learning rate per parameter (values of the original 3DGS)
  blob count   optional densification (densify.py): add blobs where detail is missing
"""

import math
import random

import torch
from torch import nn

from gaussian_slam.geometry import Camera
from gaussian_slam.metrics import psnr, ssim
from gaussian_slam.step3_optimizations.tiles import render_tiled

# Spherical harmonics constants (real SH, same order and signs as the original 3DGS code).
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)  # fmt: skip
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)  # fmt: skip


def sh_basis(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """(N, 3) unit view directions -> (N, (degree+1)^2 - 1): the SH basis functions of
    degrees 1..degree (degree 0 is the constant, handled by the base colour)."""
    x, y, z = dirs.unbind(-1)
    out = []
    if degree >= 1:
        out += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy),
            SH_C2[3] * x * z, SH_C2[4] * (xx - yy),
        ]  # fmt: skip
    if degree >= 3:
        out += [
            SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy),
        ]  # fmt: skip
    if not out:
        return dirs.new_zeros(dirs.shape[0], 0)
    return torch.stack(out, dim=-1)


class Gaussians3D(nn.Module):
    def __init__(self, n: int, extent: float = 1.0, sh_degree: int = 3):
        """n random blobs inside the cube [-extent, extent]^3.

        sh_degree: 0 = one colour per blob, as in step 2; 1-3 = the colour also changes with
        the viewing direction. fit() switches the degrees on one at a time (active_sh_degree).
        """
        super().__init__()
        scale = extent / n ** (1 / 3)  # so the blobs roughly fill the cube

        # Raw (unconstrained) parameters, as in step 2. values() maps them to valid values.
        self.means = nn.Parameter((torch.rand(n, 3) * 2 - 1) * extent)
        self.quats = nn.Parameter(torch.randn(n, 4))  # normalized inside quat_to_rotmat
        self.log_scales = nn.Parameter(torch.full((n, 3), math.log(scale)))
        self.color_raw = nn.Parameter(torch.zeros(n, 3))  # base colour (SH degree 0)
        self.sh_rest = nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3))  # degrees 1+
        self.opacity_raw = nn.Parameter(torch.zeros(n))
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # fit() starts it at 0 and raises it

    def values(self, cam: Camera | None = None):
        """(means, quats, scales, colors, opacities) as valid values, in render order.

        cam: needed for view-dependent colour; without it, only the base colour is used.
        """
        color = self.color_raw
        d = min(self.active_sh_degree, self.sh_degree)
        if cam is not None and d > 0:
            centre = -cam.R.T @ cam.t  # camera position in the world
            dirs = self.means - centre
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k = (d + 1) ** 2 - 1
            color = color + torch.einsum("nk,nkc->nc", sh_basis(dirs, d), self.sh_rest[:, :k])
        return (
            self.means,
            self.quats,
            torch.exp(self.log_scales),
            torch.sigmoid(color),
            torch.sigmoid(self.opacity_raw),
        )

    def forward(self, cam: Camera, background=None, **tile_options):
        """image, depth, alpha, info; tile_options go to tiles.render_tiled()."""
        return render_tiled(*self.values(cam), cam, background, **tile_options)


def fit(
    model: Gaussians3D,
    cams: list[Camera],
    targets: list[torch.Tensor],
    steps: int = 10_000,
    lr_means: float = 0.001,
    lr_colors: float = 0.0025,
    lr_opacity: float = 0.025,
    lr_scales: float = 0.005,
    lr_quats: float = 0.001,
    lr_sh: float | None = None,
    lr_min_ratio: float = 0.01,
    ssim_weight: float = 0.2,
    sh_every: int = 1000,
    densify=None,
    background: torch.Tensor | None = None,
    tile_options: dict | None = None,
    eval_views: int | None = None,
):
    """Fit the blobs to images from several cameras. Returns [(step, mean PSNR), ...].

    Each step uses one random camera. Every parameter has its own learning rate, and all of
    them decay along a cosine to lr_min_ratio of their start value.
    lr_sh: for the view-dependent colour coefficients (None = lr_colors / 20, as in 3DGS).
    ssim_weight: loss = (1 - w) * L1 + w * (1 - SSIM).
    sh_every: one more SH degree is switched on every this many steps, so the base colour
    settles before the view-dependent part (0 = all degrees from the start).
    densify: a densify.DensityControl, or None to keep the blob count fixed.
    background (3,): must match the colour behind the objects in the targets (white for the
    Blender dataset).
    tile_options: passed to tiles.render_tiled(), e.g. {"budget_mb": 512}.
    eval_views: the PSNR printed every 100 steps uses this many evenly spaced cameras
    (None = all of them; with 100 cameras, evaluating all would double the run time).
    """
    tile_options = tile_options or {}
    n_eval = len(cams) if eval_views is None else min(eval_views, len(cams))
    eval_ids = [round(k * (len(cams) - 1) / max(1, n_eval - 1)) for k in range(n_eval)]

    groups = [
        {"params": [model.means], "lr": lr_means},
        {"params": [model.color_raw], "lr": lr_colors},
        {"params": [model.opacity_raw], "lr": lr_opacity},
        {"params": [model.log_scales], "lr": lr_scales},
        {"params": [model.quats], "lr": lr_quats},
    ]
    if model.sh_rest.numel() > 0:
        groups.append({"params": [model.sh_rest], "lr": lr_colors / 20 if lr_sh is None else lr_sh})
    optimizer = torch.optim.Adam(groups)

    def cosine(s):  # factor on every group's starting lr: 1 -> lr_min_ratio
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * s / steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)
    model.active_sh_degree = 0 if sh_every else model.sh_degree
    history = []
    try:
        for step in range(1, steps + 1):
            if sh_every and step % sh_every == 0:
                model.active_sh_degree = min(model.active_sh_degree + 1, model.sh_degree)
            i = random.randrange(len(cams))
            image, _, _, info = model(cams[i], background, **tile_options)
            l1 = torch.mean(torch.abs(image - targets[i]))
            loss = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(image, targets[i]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if densify is not None:
                densify.update(step, model, optimizer, info)  # may add / remove blobs
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    scores = [
                        psnr(model(cams[k], background, **tile_options)[0], targets[k])
                        for k in eval_ids
                    ]
                history.append((step, sum(scores) / len(scores)))
                print(
                    f"step {step:5d}  mean PSNR {history[-1][1]:.2f} dB  "
                    f"blobs {model.means.shape[0]}  pairs {info['pairs']}  passes {info['passes']}"
                )
    except KeyboardInterrupt:
        # Ctrl+C stops training early; the caller still gets the model as it is now.
        print(f"\ninterrupted at step {step}, stopping early")
    return history
