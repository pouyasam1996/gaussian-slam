"""Step 4: snap each camera onto the fused surface using its own depth image.

Lay the camera's depth points (a small 3D scan) over the room's fused surface and find the
small turn + shift of the whole scan that makes it fit best. Where it fits, the camera was.

One iteration, for one camera:
  1. Depth pixels -> 3D points p in the world, with the camera's current pose.
  2. For each p, the nearest surface point q with its normal n (from surface.py). Pairs
     further apart than `max_dist` are ignored (probably not the same spot).
  3. Gap along the normal: r = n . (p - q). Sliding along a wall leaves r unchanged, so
     only real errors count.
  4. Move all points by a small rotation w (axis * angle) and shift s:  p' = p + w x p + s.
     The new gap is  r + (p x n) . w + n . s  -- linear in the 6 unknowns (w, s).
     Least squares over all pairs: (J^T J) x = -J^T r, with rows J = [p x n, n]: a 6x6
     system, solved exactly. A tiny damping keeps directions without grip (a camera seeing
     one flat wall can slide along it) from moving.
  5. Move the camera instead of the points: if world points move by p' = dR p + s, the
     camera that sees them unchanged has R' = R dR^T, t' = t - R dR^T s.
Repeat `iters` times: the nearest points change after each move.

Known as point-to-plane ICP (iterative closest point).
"""

from dataclasses import replace

import torch

from gaussian_slam.geometry import Camera
from gaussian_slam.step4_fusion.noise import small_rotation


def depth_points(depth: torch.Tensor, cam: Camera, sample: int = 4) -> torch.Tensor:
    """World points from every `sample`-th pixel with a depth measurement."""
    d = depth[::sample, ::sample]
    rows = torch.arange(d.shape[0], device=d.device, dtype=d.dtype) * sample
    cols = torch.arange(d.shape[1], device=d.device, dtype=d.dtype) * sample
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    ok = d > 0
    z, u, v = d[ok], u[ok], v[ok]
    p_cam = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1)
    return (p_cam - cam.t) @ cam.R  # x_world = R^T (x_cam - t), written for rows


def nearest(points: torch.Tensor, surface: torch.Tensor, chunk: int = 1024):
    """Index of, and distance to, the nearest surface point for every point (brute force)."""
    idx, dist = [], []
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], surface)
        m = d.min(dim=1)
        idx.append(m.indices)
        dist.append(m.values)
    return torch.cat(idx), torch.cat(dist)


def snap_camera(
    cam: Camera,
    depth: torch.Tensor,
    surface: torch.Tensor,
    normals: torch.Tensor,
    iters: int = 5,
    max_dist: float = 0.1,
    sample: int = 4,
    damping: float = 1e-6,
    min_pairs: int = 100,
):
    """Camera moved onto the surface; also the mean |gap| (m) before and after, and pairs used."""
    gap_before = gap_after = float("nan")
    pairs = 0
    for it in range(iters):
        p = depth_points(depth, cam, sample)
        idx, dist = nearest(p, surface)
        keep = dist < max_dist
        pairs = int(keep.sum())
        if pairs < min_pairs:
            break
        p, q, n = p[keep], surface[idx[keep]], normals[idx[keep]]
        r = ((p - q) * n).sum(-1)  # gap along the normal
        if it == 0:
            gap_before = r.abs().mean().item()

        J = torch.cat([torch.linalg.cross(p, n), n], dim=-1)  # (P, 6)
        A = (J.T @ J).cpu().double()  # the Apple GPU has no float64; the 6x6 solve is tiny
        b = -(J.T @ r).cpu().double()
        A = A + damping * A.diagonal().mean() * torch.eye(6, dtype=A.dtype)
        x = torch.linalg.solve(A, b).float().to(cam.R.device)

        dR, s = small_rotation(x[:3].cpu()).to(cam.R), x[3:]
        R_new = cam.R @ dR.T
        cam = replace(cam, R=R_new, t=cam.t - R_new @ s)

    p = depth_points(depth, cam, sample)
    idx, dist = nearest(p, surface)
    keep = dist < max_dist
    if keep.any():
        gap_after = (
            (((p[keep] - surface[idx[keep]]) * normals[idx[keep]]).sum(-1)).abs().mean().item()
        )
    return cam, gap_before, gap_after, pairs


def snap_cameras(cams, depths, surface, normals, **options):
    """snap_camera() for every camera. Returns new cameras and (mean gap before, after) in mm."""
    new, before, after = [], [], []
    for cam, depth in zip(cams, depths, strict=True):
        c, b, a, _ = snap_camera(cam, depth, surface, normals, **options)
        new.append(c)
        if b == b and a == a:  # skip cameras that found too few pairs (NaN)
            before.append(b)
            after.append(a)

    def mean_mm(x):
        return sum(x) / max(len(x), 1) * 1000

    return new, mean_mm(before), mean_mm(after)
