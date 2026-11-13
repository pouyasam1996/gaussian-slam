"""Step 3: fit blobs to the Blender Lego scene, then test on unseen views.

uv run python scripts/step3_optimizations/optimized_3dgaussian_run.py               # 100 px, 10k steps
uv run python scripts/step3_optimizations/optimized_3dgaussian_run.py --steps 3000  # quicker check

Starts from random blobs in a cube (this dataset has no point cloud), white background,
tiled renderer, densification on, and the original 3DGS recipe: a learning rate per
parameter, view-dependent colour (SH degree 3), loss 0.8 L1 + 0.2 (1 - SSIM), opacity reset
every 3000 steps.

Before training, it checks that the tiled renderer draws the same image, with the same
gradients, as step 2's dense render(); a mismatch stops the script with an AssertionError.

Outputs in outputs/lego/:
  test_views.png  6 test views: target / fitted / error
  training.png    PSNR on a few training views during training
  orbit.gif       a full turn around the fitted model
  model.pt        the fitted blobs (model.state_dict())
"""

import argparse
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gaussian_slam.device import get_device
from gaussian_slam.geometry import look_at
from gaussian_slam.metrics import psnr, ssim
from gaussian_slam.step2_3d.gaussians3d import render
from gaussian_slam.step3_optimizations.blender import load_split
from gaussian_slam.step3_optimizations.densify import DensityControl
from gaussian_slam.step3_optimizations.gaussians3d import Gaussians3D, fit
from gaussian_slam.step3_optimizations.tiles import render_tiled


def check_tiled():
    """Step 2's dense render() and the tiled renderer draw the same picture, with the same
    gradients.

    The image is 40x40 (not a multiple of 16) and buckets are tiny, so a tile needs several
    passes. Early stopping is off, so the only difference is the 1/255 cut-off.
    """
    torch.manual_seed(1)
    n = 200
    cam = look_at((0.0, -4.0, 1.0), (0.0, 0.0, 0.0), 40, 40)
    blobs = [
        torch.rand(n, 3) * 2 - 1,
        torch.randn(n, 4),
        torch.rand(n, 3) * 0.15 + 0.02,
        torch.rand(n, 3),
        torch.rand(n) * 0.9 + 0.05,
    ]
    blobs = [x.requires_grad_() for x in blobs]
    dense = render(*blobs, cam)
    tiled = render_tiled(*blobs, cam, buckets=(4, 8), light_eps=0.0)
    image_diff = (dense[0] - tiled[0]).abs().max().item()
    grads = [torch.autograd.grad(r[0].sum(), blobs[0])[0] for r in (dense, tiled)]
    grad_diff = ((grads[0] - grads[1]).norm() / grads[0].norm()).item()
    print(
        f"check: dense vs tiled, max pixel diff {image_diff:.4f}, "
        f"relative gradient diff {grad_diff:.4f}"
    )
    assert image_diff < 0.05 and grad_diff < 0.05, "tiled renderer does not match dense"


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/lego", help="scene folder")
parser.add_argument("--size", type=int, default=100, help="image side in pixels (data: 800)")
parser.add_argument("--n", type=int, default=10_000, help="blobs at the start")
parser.add_argument("--steps", type=int, default=10_000)
parser.add_argument("--no-densify", action="store_true")
parser.add_argument("--max-blobs", type=int, default=30_000)
# 2.5e-6 suited the old MSE loss; L1 + SSIM pulls harder, so this is a first guess.
parser.add_argument("--grad-threshold", type=float, default=7.5e-6, help="set near the printed p90")
parser.add_argument("--reset-every", type=int, default=3000, help="opacity reset period, 0 = never")
parser.add_argument(
    "--small-px", type=float, default=1.5, help="blobs smaller than this many pixels clone"
)
parser.add_argument(
    "--budget-mb", type=float, default=512, help="memory for blending one group of tiles"
)
parser.add_argument("--extent", type=float, default=1.3, help="half-width of the start cube")
parser.add_argument("--eval-views", type=int, default=5, help="views for the PSNR during training")
parser.add_argument("--lr-means", type=float, default=0.001)
parser.add_argument("--lr-colors", type=float, default=0.0025)
parser.add_argument("--lr-opacity", type=float, default=0.025)
parser.add_argument("--lr-scales", type=float, default=0.005)
parser.add_argument("--lr-quats", type=float, default=0.001)
parser.add_argument("--sh-degree", type=int, default=3, help="view-dependent colour, 0 = off")
parser.add_argument(
    "--ssim-weight", type=float, default=0.2, help="loss (1-w) L1 + w (1-SSIM); 0 = plain L1"
)
args = parser.parse_args()

check_tiled()
torch.manual_seed(0)
random.seed(0)
device = get_device()
out = Path("outputs/lego")
out.mkdir(parents=True, exist_ok=True)

# 1. Data: all 100 training views (each step uses one, so more views cost nothing per step),
# all 200 test views for the final score.
t0 = time.time()
train_cams, train_images = load_split(args.data, "train", args.size)
test_cams, test_images = load_split(args.data, "test", args.size)
train_cams = [c.to(device) for c in train_cams]
test_cams = [c.to(device) for c in test_cams]
train_images = [im.to(device) for im in train_images]
test_images = [im.to(device) for im in test_images]
white = torch.ones(3, device=device)
print(
    f"loaded {len(train_cams)} train + {len(test_cams)} test views at {args.size} px "
    f"in {time.time() - t0:.1f} s, device {device}"
)

# 2. Model and densification.
model = Gaussians3D(args.n, extent=args.extent, sh_degree=args.sh_degree).to(device)
densify = None
if not args.no_densify:
    # "Small" in pixels: one pixel at the object's distance covers distance / fx world units.
    # Camera centre = -R^T t; the object sits at the origin.
    distance = np.mean([(-(c.R.T @ c.t)).norm().item() for c in train_cams])
    pixel = distance / train_cams[0].fx
    print(
        f"one pixel ~ {pixel:.4f} units at distance {distance:.2f}; "
        f"clone below {args.small_px} px = {args.small_px * pixel:.4f} units"
    )
    densify = DensityControl(
        extent=args.extent,
        grad_threshold=args.grad_threshold,
        stop=int(0.6 * args.steps),
        reset_every=args.reset_every,
        max_blobs=args.max_blobs,
        small=args.small_px * pixel / args.extent,  # DensityControl wants a share of extent
    )
options = {"budget_mb": args.budget_mb}  # passed to the tiled renderer

# 3. Train.
t0 = time.time()
history = fit(
    model,
    train_cams,
    train_images,
    steps=args.steps,
    densify=densify,
    background=white,
    tile_options=options,
    eval_views=args.eval_views,
    lr_means=args.lr_means,
    lr_colors=args.lr_colors,
    lr_opacity=args.lr_opacity,
    lr_scales=args.lr_scales,
    lr_quats=args.lr_quats,
    ssim_weight=args.ssim_weight,
)
train_time = time.time() - t0
torch.save(model.state_dict(), out / "model.pt")

# 4. Final scores: every view.


@torch.no_grad()
def render_all(cams):
    return [model(c, white, **options)[0].clamp(0, 1) for c in cams]


test_renders = render_all(test_cams)
test_scores = [psnr(r, t) for r, t in zip(test_renders, test_images, strict=True)]
test_ssim = [ssim(r, t).item() for r, t in zip(test_renders, test_images, strict=True)]
train_scores = [psnr(r, t) for r, t in zip(render_all(train_cams), train_images, strict=True)]
print(
    f"\ntrained {args.steps} steps in {train_time / 60:.1f} min "
    f"({1000 * train_time / args.steps:.0f} ms/step), {model.means.shape[0]} blobs\n"
    f"PSNR  train {np.mean(train_scores):.2f} dB   test (200 unseen views) "
    f"{np.mean(test_scores):.2f} dB   test SSIM {np.mean(test_ssim):.3f}"
)


def to_numpy(image):
    return image.detach().cpu().numpy()


# 5. Pictures.
picks = [round(k * (len(test_cams) - 1) / 5) for k in range(6)]
fig, axes = plt.subplots(3, len(picks), figsize=(2.4 * len(picks), 7.4))
for j, i in enumerate(picks):
    axes[0, j].imshow(to_numpy(test_images[i]))
    axes[1, j].imshow(to_numpy(test_renders[i]))
    axes[1, j].set_xlabel(f"{test_scores[i]:.1f} dB")
    err = (test_renders[i] - test_images[i]).abs().mean(-1)
    axes[2, j].imshow(to_numpy(err), cmap="magma", vmin=0, vmax=0.3)
    axes[0, j].set_title(f"test view {i}")
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
for r, name in enumerate(["target", "fitted", "|fitted - target|"]):
    axes[r, 0].set_ylabel(name, fontsize=11)
fig.suptitle(f"Lego at {args.size} px, test views never seen in training")
fig.tight_layout()
fig.savefig(out / "test_views.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
steps, scores = zip(*history, strict=True)
ax.plot(steps, scores, label=f"train ({args.eval_views} views)")
ax.axhline(np.mean(test_scores), color="tab:red", ls="--", label="test, final")
ax.set_xlabel("step")
ax.set_ylabel("PSNR (dB)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "training.png", dpi=120)
plt.close(fig)

# A full turn at the dataset's distance (~4 units) and field of view, 30 degrees above the
# table, rendered at 2x the training size (the blobs are a 3D scene, any size works).
size = 2 * args.size
fov = math.degrees(2 * math.atan(0.5 * args.size / train_cams[0].fx))
frames = []
with torch.no_grad():
    for k in range(60):
        a = 2 * math.pi * k / 60
        eye = (3.5 * math.cos(a), 3.5 * math.sin(a), 2.0)
        cam = look_at(eye, (0.0, 0.0, 0.0), size, size, fov_deg=fov).to(device)
        image = model(cam, white, **options)[0].clamp(0, 1)
        frames.append(Image.fromarray((to_numpy(image) * 255).astype(np.uint8)))
frames[0].save(out / "orbit.gif", save_all=True, append_images=frames[1:], duration=60, loop=0)
print(f"saved test_views.png, training.png, orbit.gif, model.pt to {out}")
