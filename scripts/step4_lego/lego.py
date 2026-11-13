"""Step 4: fit blobs to the Blender Lego scene, then test on unseen views.

uv run python scripts/step4_lego/lego.py                                     # fast test: 100 px, ~5-10 min
uv run python scripts/step4_lego/lego.py --size 200 --steps 10000 --max-blobs 50000   # better, ~1 h
uv run python scripts/step4_lego/lego.py --mode exact                        # exact training steps only

Starts from random blobs in a cube (this dataset has no point cloud), white background,
"sampled" training steps with an exact step every 10, densification on.

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
from gaussian_slam.metrics import psnr
from gaussian_slam.step3_3d_advanced.densify import DensityControl
from gaussian_slam.step3_3d_advanced.gaussians3d import Gaussians3D, fit
from gaussian_slam.step3_3d_advanced.memory import MemoryBudget
from gaussian_slam.step4_lego.blender import load_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/lego", help="scene folder")
parser.add_argument("--size", type=int, default=100, help="image side in pixels (data: 800)")
parser.add_argument("--n", type=int, default=5000, help="blobs at the start")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--mode", choices=["dense", "exact", "sampled", "auto"], default="sampled")
parser.add_argument("--exact-every", type=int, default=10, help="0 = never an exact step")
parser.add_argument("--keep-ratio", type=float, nargs=2, default=(0.3, 0.9), help="start end")
parser.add_argument("--no-densify", action="store_true")
parser.add_argument("--max-blobs", type=int, default=20_000)
parser.add_argument("--grad-threshold", type=float, default=2e-4)
parser.add_argument("--no-memory", action="store_true", help="fixed budget_mb, no adjusting")
parser.add_argument("--extent", type=float, default=1.3, help="half-width of the start cube")
parser.add_argument("--eval-views", type=int, default=5, help="views for the PSNR during training")
args = parser.parse_args()

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

# 2. Model, densification, memory budget.
model = Gaussians3D(args.n, extent=args.extent).to(device)
densify = None
if not args.no_densify:
    densify = DensityControl(
        extent=args.extent,
        grad_threshold=args.grad_threshold,
        stop=int(0.6 * args.steps),
        max_blobs=args.max_blobs,
    )
memory = None if args.no_memory or device.type == "cpu" else MemoryBudget(device)

# 3. Train.
t0 = time.time()
history = fit(
    model,
    train_cams,
    train_images,
    steps=args.steps,
    mode=args.mode,
    exact_every=args.exact_every if args.mode in ("sampled", "auto") else 0,
    densify=densify,
    background=white,
    keep_ratio_schedule=tuple(args.keep_ratio) if args.mode in ("sampled", "auto") else None,
    memory=memory,
    eval_views=args.eval_views,
)
train_time = time.time() - t0
torch.save(model.state_dict(), out / "model.pt")

# 4. Final scores: every view, exact renderer.
eval_mode = "dense" if args.mode == "dense" else "exact"
options = {"budget_mb": memory.budget_mb} if memory is not None else {}


@torch.no_grad()
def render_all(cams):
    return [model(c, white, mode=eval_mode, **options)[0].clamp(0, 1) for c in cams]


test_renders = render_all(test_cams)
test_scores = [psnr(r, t) for r, t in zip(test_renders, test_images, strict=True)]
train_scores = [psnr(r, t) for r, t in zip(render_all(train_cams), train_images, strict=True)]
print(
    f"\ntrained {args.steps} steps in {train_time / 60:.1f} min "
    f"({1000 * train_time / args.steps:.0f} ms/step), {model.means.shape[0]} blobs\n"
    f"PSNR  train {np.mean(train_scores):.2f} dB   test (200 unseen views) "
    f"{np.mean(test_scores):.2f} dB"
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
        image = model(cam, white, mode=eval_mode, **options)[0].clamp(0, 1)
        frames.append(Image.fromarray((to_numpy(image) * 255).astype(np.uint8)))
frames[0].save(out / "orbit.gif", save_all=True, append_images=frames[1:], duration=60, loop=0)
print(f"saved test_views.png, training.png, orbit.gif, model.pt to {out}")
