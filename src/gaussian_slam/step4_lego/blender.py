"""Step 4: load the NeRF-Synthetic (Blender) dataset at a chosen image size.

Each scene folder has transforms_{train,val,test}.json and the PNG images they list:
  camera_angle_x      horizontal field of view in radians (the same for every frame)
  frames[i].file_path  image path without ".png", e.g. "./train/r_0"
  frames[i].transform_matrix  4x4 camera -> world pose, Blender/OpenGL convention

The images are RGBA (colour + transparency): the object over a transparent background.
They are placed on white, then shrunk by averaging blocks of pixels.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_slam.geometry import Camera, opengl_to_opencv


def load_split(folder, split: str, size: int, every: int = 1):
    """Cameras and images (size, size, 3) in [0, 1] for transforms_<split>.json.

    every: keep every `every`-th frame (1 = all).
    """
    folder = Path(folder)
    meta = json.loads((folder / f"transforms_{split}.json").read_text())
    # Pinhole focal length from the field of view: half the image width covers half the angle.
    f = 0.5 * size / math.tan(meta["camera_angle_x"] / 2)

    cams, images = [], []
    for frame in meta["frames"][::every]:
        R, t = opengl_to_opencv(torch.tensor(frame["transform_matrix"], dtype=torch.float32))
        cams.append(Camera(R, t, f, f, size / 2, size / 2, size, size))

        path = folder / frame["file_path"]
        if path.suffix != ".png":
            path = path.with_name(path.name + ".png")
        rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255
        rgb, alpha = rgba[..., :3], rgba[..., 3:]
        on_white = torch.from_numpy(rgb * alpha + (1 - alpha))  # (800, 800, 3)
        # Shrink by averaging (antialiased); composite first, so edges don't turn dark.
        small = F.interpolate(on_white.permute(2, 0, 1)[None], size=(size, size), mode="area")
        images.append(small[0].permute(1, 2, 0).contiguous())
    return cams, images
