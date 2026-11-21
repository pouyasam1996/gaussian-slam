"""Step 4: load a Replica RGB-D scene (the release used by Gaussian SLAM papers).

Scene folder layout, e.g. data/Replica/room0:
  results/frame000000.jpg  colour, 1200 x 680
  results/depth000000.png  depth, 16-bit; metres = value / 6553.5 (0 = no measurement)
  traj.txt                 one line per frame: 16 numbers, a 4x4 camera -> world pose,
                           already in OpenCV axes (x right, y down, z forward)

Every frame uses the same pinhole camera: fx = fy = 600, cx = 599.5, cy = 339.5.
These are the values of the common Replica release; check them against your download.

Frames are read one at a time (a generator): 2000 full-size frames would not fit in memory.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_slam.geometry import Camera

WIDTH, HEIGHT = 1200, 680
FX = FY = 600.0
CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5


def load_poses(folder) -> torch.Tensor:
    """(N, 4, 4) camera -> world poses from traj.txt."""
    rows = np.loadtxt(Path(folder) / "traj.txt", dtype=np.float64)
    return torch.from_numpy(rows.reshape(-1, 4, 4)).float()


def pose_to_camera(c2w: torch.Tensor, stride: int = 1) -> Camera:
    """Camera -> world pose -> our Camera (world -> camera), for images shrunk by `stride`.

    Keeping every stride-th pixel maps old pixel u to new pixel u / stride, so the focal
    length and the image centre are simply divided by stride.
    """
    R = c2w[:3, :3].T  # inverse of a rotation is its transpose
    t = -R @ c2w[:3, 3]
    s = stride
    w, h = (WIDTH + s - 1) // s, (HEIGHT + s - 1) // s  # sizes of image[::s, ::s]
    return Camera(R, t, FX / s, FY / s, CX / s, CY / s, w, h)


def load_frame(folder, index: int, stride: int = 1):
    """Depth (H, W) in metres and colour (H, W, 3) in [0, 1] for one frame."""
    results = Path(folder) / "results"
    depth = np.asarray(Image.open(results / f"depth{index:06d}.png"), dtype=np.float32)
    color = np.asarray(Image.open(results / f"frame{index:06d}.jpg").convert("RGB"))
    depth = torch.from_numpy(depth[::stride, ::stride] / DEPTH_SCALE)
    color = torch.from_numpy(color[::stride, ::stride].astype(np.float32) / 255)
    return depth.contiguous(), color.contiguous()


def frames(folder, every: int = 5, stride: int = 1, limit: int | None = None):
    """Yield (index, camera, depth, colour) for every `every`-th frame, at most `limit` of them."""
    poses = load_poses(folder)
    indices = range(0, len(poses), every)
    if limit is not None:
        indices = indices[:limit]
    for i in indices:
        depth, color = load_frame(folder, i, stride)
        yield i, pose_to_camera(poses[i], stride), depth, color
