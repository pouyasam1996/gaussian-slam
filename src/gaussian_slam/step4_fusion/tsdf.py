"""Step 4: fuse depth images into a sparse TSDF grid (truncated signed distance).

The idea: cut space into small cubes (voxels). Each voxel stores how far it is from the
nearest surface, measured along the camera's viewing axis z:
  positive = in front of the surface (free space), negative = behind it (inside)
Distances are clipped to +-trunc and divided by trunc, so every value is in [-1, 1].
Each new depth image updates a running average, so the noise of single frames cancels out.
The surface is where the value crosses zero (see surface.py).

Only space near surfaces is stored: voxels come in blocks of B x B x B, and a block is
created the first time a depth measurement lands near it. Blocks are found by a sorted
list of integer codes (one code per block) and binary search.

The update rules copy Open3D's VoxelBlockGrid CPU kernel, so both give the same surface
(on room0, every 20th frame, the two surfaces agree to within 0.01 mm on average):
  - voxel (i, j, k) sits at world position (i, j, k) * voxel_size
  - depth is read at one pixel, no blending of neighbours
  - sdf = measured depth - voxel depth along z;  skipped if sdf < -trunc
  - value = (weight * value + new) / (weight + 1);  weight += 1, no cap
"""

import math

import torch

from gaussian_slam.geometry import Camera

_OFF = 1 << 20  # shift coordinates to be positive; 21 bits each fit into one int64


def encode(ijk: torch.Tensor) -> torch.Tensor:
    """(N, 3) integer coordinates -> (N,) int64 codes, one per coordinate."""
    x = ijk + _OFF
    return (x[:, 0] << 42) | (x[:, 1] << 21) | x[:, 2]


def find(sorted_codes: torch.Tensor, query: torch.Tensor):
    """Positions of `query` codes in `sorted_codes`, and whether each one was found."""
    if len(sorted_codes) == 0:
        zeros = torch.zeros_like(query)
        return zeros, zeros.bool()
    idx = torch.searchsorted(sorted_codes, query).clamp(max=len(sorted_codes) - 1)
    return idx, sorted_codes[idx] == query


class TSDFVolume:
    def __init__(
        self,
        voxel_size: float = 0.02,
        block: int = 8,
        trunc_voxels: float = 4.0,
        depth_max: float = 10.0,
        device="cpu",
    ):
        self.voxel_size = voxel_size
        self.block = block
        self.trunc = trunc_voxels * voxel_size  # in metres
        self.depth_max = depth_max
        self.device = torch.device(device)
        b3 = block**3
        r = torch.arange(block, device=self.device)
        # (B^3, 3) offsets of the voxels inside one block
        self.local = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        # One row per block, rows kept sorted by code.
        self.codes = torch.empty(0, dtype=torch.long, device=self.device)
        self.keys = torch.empty(0, 3, dtype=torch.long, device=self.device)  # block coordinates
        self.tsdf = torch.empty(0, b3, device=self.device)
        self.weight = torch.empty(0, b3, device=self.device)
        self.color = torch.empty(0, b3, 3, device=self.device)

    def touched_blocks(self, depth: torch.Tensor, cam: Camera, down: int = 4) -> torch.Tensor:
        """(M, 3) blocks within +-trunc of the measured surface, from every `down`-th pixel.

        Each pixel is lifted to 3D at a few depths between depth - trunc and depth + trunc
        (every half block), and the blocks containing those points are kept.
        """
        d = depth[::down, ::down]
        rows = torch.arange(d.shape[0], device=self.device) * down
        cols = torch.arange(d.shape[1], device=self.device) * down
        v, u = torch.meshgrid(rows.float(), cols.float(), indexing="ij")
        ok = (d > 0) & (d <= self.depth_max)
        d, u, v = d[ok], u[ok], v[ok]

        steps = math.ceil(2 * self.trunc / (self.voxel_size * self.block / 2)) + 1
        s = torch.linspace(-1, 1, steps, device=self.device)
        z = (d[:, None] + s * self.trunc).reshape(-1)
        u = u[:, None].expand(-1, steps).reshape(-1)
        v = v[:, None].expand(-1, steps).reshape(-1)
        keep = z > 0
        z, u, v = z[keep], u[keep], v[keep]

        # Pixel -> camera frame -> world frame: x_world = R^T (x_cam - t), written for rows.
        p_cam = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1)
        p_world = (p_cam - cam.t) @ cam.R
        keys = torch.floor(p_world / (self.voxel_size * self.block)).long()
        return torch.unique(keys, dim=0)

    def _allocate(self, keys: torch.Tensor) -> torch.Tensor:
        """Create the blocks that don't exist yet; return the row of every key."""
        codes = encode(keys)
        _, found = find(self.codes, codes)
        new = ~found
        if new.any():
            n, b3 = int(new.sum()), self.block**3
            self.codes = torch.cat([self.codes, codes[new]])
            self.keys = torch.cat([self.keys, keys[new]])
            self.tsdf = torch.cat([self.tsdf, self.tsdf.new_zeros(n, b3)])
            self.weight = torch.cat([self.weight, self.weight.new_zeros(n, b3)])
            self.color = torch.cat([self.color, self.color.new_zeros(n, b3, 3)])
            order = torch.argsort(self.codes)
            self.codes, self.keys = self.codes[order], self.keys[order]
            self.tsdf, self.weight, self.color = (
                self.tsdf[order],
                self.weight[order],
                self.color[order],
            )
        idx, _ = find(self.codes, codes)
        return idx

    @torch.no_grad()
    def integrate(self, depth: torch.Tensor, color: torch.Tensor, cam: Camera):
        """Fuse one frame: depth (H, W) in metres, colour (H, W, 3) in [0, 1]."""
        depth, color, cam = depth.to(self.device), color.to(self.device), cam.to(self.device)
        idx = self._allocate(self.touched_blocks(depth, cam))
        b3 = self.block**3

        # Every voxel of the touched blocks: integer coordinates and its row in the storage.
        ijk = (self.keys[idx][:, None, :] * self.block + self.local).reshape(-1, 3)
        flat = (idx[:, None] * b3 + torch.arange(b3, device=self.device)).reshape(-1)

        # Voxel -> camera frame -> pixel.
        p_cam = (ijk.float() * self.voxel_size) @ cam.R.T + cam.t
        x, y, z = p_cam.unbind(-1)
        zs = z.clamp(min=1e-6)  # avoids dividing by 0; those voxels are dropped below
        u = cam.fx * x / zs + cam.cx
        v = cam.fy * y / zs + cam.cy
        H, W = depth.shape
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u <= W - 1) & (v <= H - 1)
        flat, z, ui, vi = flat[ok], z[ok], u[ok].long(), v[ok].long()  # .long() rounds down here

        # Signed distance along z, kept only near and in front of the surface.
        d = depth[vi, ui]
        sdf = d - z
        ok = (d > 0) & (d <= self.depth_max) & (sdf >= -self.trunc)
        flat, sdf, ui, vi = flat[ok], sdf[ok], ui[ok], vi[ok]
        sdf = sdf.clamp(max=self.trunc) / self.trunc

        # Running averages; views share memory with the storage, so this updates it.
        tsdf, weight, rgb = self.tsdf.view(-1), self.weight.view(-1), self.color.view(-1, 3)
        w = weight[flat]
        tsdf[flat] = (w * tsdf[flat] + sdf) / (w + 1)
        rgb[flat] = (w[:, None] * rgb[flat] + color[vi, ui]) / (w[:, None] + 1)
        weight[flat] = w + 1

    def voxels(self, min_weight: float = 0.0):
        """Voxels seen by more than `min_weight` frames: (N, 3) coordinates, tsdf, weight, colour."""
        ijk = (self.keys[:, None, :] * self.block + self.local).reshape(-1, 3)
        keep = self.weight.view(-1) > min_weight
        return (
            ijk[keep],
            self.tsdf.view(-1)[keep],
            self.weight.view(-1)[keep],
            self.color.view(-1, 3)[keep],
        )

    def memory_mb(self) -> float:
        """Storage used by tsdf + weight + colour (5 floats per voxel)."""
        return len(self.codes) * self.block**3 * 5 * 4 / 1e6
