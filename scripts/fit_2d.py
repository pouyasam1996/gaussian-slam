"""Step 1.2: fit 2D Gaussians to a photo.

uv run python scripts/fit_2d.py path/to/photo.jpg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gaussian_slam.device import get_device
from gaussian_slam.gaussians2d import Gaussians2D, fit

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--size", type=int, default=256, help="longer side in pixels")
parser.add_argument("--n", type=int, default=2000, help="number of blobs")
parser.add_argument("--steps", type=int, default=2000, help="number of optimization steps")
args = parser.parse_args()

torch.manual_seed(0)
device = get_device()
image = Image.open(args.image).convert("RGB")
image.thumbnail((args.size, args.size))  # shrink in place, keeps the aspect ratio
target = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).to(device)  # (H, W, 3)

height, width = target.shape[:2]
model = Gaussians2D(args.n, height, width).to(device)
history = fit(model, target, steps=args.steps)

out = Path("outputs/fit2d")
out.mkdir(parents=True, exist_ok=True)
image.save(out / "target.png")
with torch.no_grad():
    final = model().clamp(0, 1).cpu().numpy()
Image.fromarray((final * 255).round().astype(np.uint8)).save(out / "final.png")

if history:  # empty if stopped with Ctrl+C before the first log line
    steps, values = zip(*history, strict=True)
    plt.plot(steps, values)
    plt.xlabel("step")
    plt.ylabel("PSNR (dB)")
    plt.savefig(out / "psnr.png")
print(f"saved to {out}")
