"""Step 4: surface points from a TSDF grid, and a small PLY writer.

The surface lies where the stored distance changes sign between two neighbouring voxels.
Along each axis, voxel o (value a) and its neighbour i (value b) with a * b < 0 give one
point, placed where the straight line between a and b reaches 0:
  point = o + r * axis,   r = a / (a - b)   (0 at o, 1 at i)
Colour and normal are blended with the same r.

The normal is the direction in which the distance grows fastest (towards free space),
from central differences: n_x = value(x + 1) - value(x - 1), and the same for y and z.

Same rules as Open3D's extract_point_cloud: only voxels seen by more than `min_weight`
frames take part.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_slam.step4_fusion.tsdf import TSDFVolume, encode, find


def extract_points(vol: TSDFVolume, min_weight: float = 3.0):
    """(P, 3) points, (P, 3) colours in [0, 1], (P, 3) unit normals."""
    ijk, tsdf, _, color = vol.voxels(min_weight)
    codes = encode(ijk)
    order = torch.argsort(codes)
    codes, ijk, tsdf, color = codes[order], ijk[order], tsdf[order], color[order]
    axes = torch.eye(3, dtype=torch.long, device=ijk.device)

    def neighbour(offset):
        return find(codes, encode(ijk + offset))

    grad = torch.zeros(len(ijk), 3, device=ijk.device)
    for a in range(3):
        ip, fp = neighbour(axes[a])
        im, fm = neighbour(-axes[a])
        grad[:, a] = torch.where(fp & fm, tsdf[ip] - tsdf[im], torch.zeros_like(tsdf))

    points, colors, normals = [], [], []
    for a in range(3):
        i, found = neighbour(axes[a])
        cross = found & (tsdf * tsdf[i] < 0)
        o, i = cross.nonzero().squeeze(1), i[cross]
        r = (tsdf[o] / (tsdf[o] - tsdf[i]))[:, None]
        points.append((ijk[o].float() + r * axes[a].float()) * vol.voxel_size)
        colors.append((1 - r) * color[o] + r * color[i])
        normals.append((1 - r) * grad[o] + r * grad[i])
    return torch.cat(points), torch.cat(colors), F.normalize(torch.cat(normals), dim=-1)


def save_ply(path, points, colors=None, normals=None):
    """Write points (and colours in [0, 1], normals) as a binary PLY file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["property float x", "property float y", "property float z"]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        header += ["property float nx", "property float ny", "property float nz"]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]

    data = np.empty(len(points), dtype=fields)
    p = points.cpu().numpy()
    data["x"], data["y"], data["z"] = p[:, 0], p[:, 1], p[:, 2]
    if normals is not None:
        n = normals.cpu().numpy()
        data["nx"], data["ny"], data["nz"] = n[:, 0], n[:, 1], n[:, 2]
    if colors is not None:
        c = (colors.clamp(0, 1) * 255).round().byte().cpu().numpy()
        data["red"], data["green"], data["blue"] = c[:, 0], c[:, 1], c[:, 2]

    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    lines += [*header, "end_header"]
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(data.tobytes())
