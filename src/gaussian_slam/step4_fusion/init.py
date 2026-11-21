"""Step 4, Stage B: turn fused surface points into step 3 Gaussians (flat coins on the surface).

For every voxel that holds surface points, one blob:
  position  the average surface point in that voxel
  rotation  turns the blob's own z axis onto the surface normal, so its thin side faces out
  size      `width` voxels across the surface (neighbours overlap), `thickness` voxels thick
  colour    the fused colour; the model applies a sigmoid, so it stores logit(colour)
  opacity   `opacity`, stored as logit(opacity) for the same reason

The result is an ordinary Gaussians3D, so step 3's renderer, fit() and densification work
on it unchanged.
"""

import math

import torch
import torch.nn.functional as F

from gaussian_slam.step3_optimizations.gaussians3d import Gaussians3D


def one_per_voxel(points, colors, normals, voxel: float):
    """Average the points (and colours, normals) that fall into the same voxel."""
    keys = torch.floor(points / voxel).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n = int(inverse.max()) + 1
    ones = torch.ones(len(points), device=points.device)
    count = torch.zeros(n, device=points.device).index_add_(0, inverse, ones)

    def mean(x):
        return (
            torch.zeros(n, x.shape[1], device=x.device).index_add_(0, inverse, x) / count[:, None]
        )

    return mean(points), mean(colors), F.normalize(mean(normals), dim=-1)


def normal_to_quat(normals: torch.Tensor) -> torch.Tensor:
    """(N, 3) unit normals -> (N, 4) quaternions (w, x, y, z) turning the z axis onto them.

    The shortest turn from a to b is q = (1 + a.b, a x b), normalized. With a = z = (0, 0, 1):
    a.b = n_z and a x b = (-n_y, n_x, 0). For n = -z that is 0; any half-turn about x works.
    """
    nx, ny, nz = normals.unbind(-1)
    q = torch.stack([1 + nz, -ny, nx, torch.zeros_like(nz)], dim=-1)
    flipped = (1 + nz) < 1e-6
    q[flipped] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=q.device)
    return F.normalize(q, dim=-1)


def from_surface(
    points,
    colors,
    normals,
    voxel: float,
    width: float = 0.5,
    thickness: float = 0.1,
    opacity: float = 0.9,
    sh_degree: int = 0,
) -> Gaussians3D:
    """One flat blob per surface voxel. width / thickness are in voxels."""
    points, colors, normals = one_per_voxel(points, colors, normals, voxel)
    n = len(points)
    model = Gaussians3D(n, sh_degree=sh_degree)
    scales = torch.tensor([width, width, thickness]) * voxel
    with torch.no_grad():
        model.means.copy_(points)
        model.quats.copy_(normal_to_quat(normals))
        model.log_scales.copy_(torch.log(scales).expand(n, 3))
        model.color_raw.copy_(torch.logit(colors.clamp(0.02, 0.98)))
        model.opacity_raw.fill_(math.log(opacity / (1 - opacity)))
    return model
