"""Step 1.5: render a 3D blob scene, scramble it, and optimize it back.

uv run python scripts/synthetic_3d.py                  # a rainbow torus
uv run python scripts/synthetic_3d.py --shape helix
uv run python scripts/synthetic_3d.py --shape random   # blobs scattered in a cube

Outputs in outputs/synthetic3d/:
  scene_3d.png  blob centres in 3D: true shape, scrambled start, fitted result
  views.png     rendered images: target / start / fitted / error / depth, per camera
  training.png  mean PSNR over training steps
  heldout.png   the 2 held-out cameras alone, true vs fitted, rendered at 3x resolution

Before fitting it runs three quick checks (step 1.3 and the gradient check);
any failure stops the script with an AssertionError.
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

from gaussian_slam.device import get_device
from gaussian_slam.gaussians3d import Gaussians3D, fit, render
from gaussian_slam.geometry import Camera, look_at, opengl_to_opencv, project
from gaussian_slam.metrics import psnr


def check_projection():
    """A point projected by hand."""
    cam = Camera(torch.eye(3), torch.zeros(3), 100.0, 100.0, 32.0, 32.0, 64, 64)
    uv = project(torch.tensor([[0.5, -0.25, 2.0]]), cam)
    # u = 100 * 0.5 / 2 + 32 = 57,  v = 100 * (-0.25) / 2 + 32 = 19.5
    assert torch.allclose(uv, torch.tensor([[57.0, 19.5]])), uv


def check_cameras():
    """look_at puts its target at the image centre, and so does a converted Blender camera."""
    cam = look_at((3.0, -2.0, 1.5), (0.2, 0.1, -0.3), 64, 48)
    p = torch.tensor([[0.2, 0.1, -0.3]]) @ cam.R.T + cam.t
    assert torch.allclose(project(p, cam), torch.tensor([[32.0, 24.0]]), atol=1e-4)

    # Blender camera at z = 4 with no rotation: it looks down -z, i.e. at the origin.
    c2w = torch.eye(4)
    c2w[2, 3] = 4.0
    R, t = opengl_to_opencv(c2w)
    assert torch.allclose(t, torch.tensor([0.0, 0.0, 4.0]))  # origin is 4 units in front
    # A point above the origin (+y in Blender) must appear above the centre (smaller v).
    p = torch.tensor([[0.0, 1.0, 0.0]]) @ R.T + t
    assert p[0, 1] < 0, p


def check_gradients():
    """Automatic gradients vs small-step estimates, on the CPU in float64."""
    dtype = torch.float64
    cam = look_at((0.0, -4.0, 0.0), (0.0, 0.0, 0.0), 8, 8).to(dtype)
    inputs = (
        torch.tensor([[0.0, -1.0, 0.0], [0.3, 0.0, 0.2], [-0.2, 1.0, -0.1]]),  # depths 3, 4, 5
        torch.randn(3, 4),
        torch.full((3, 3), 0.5) + 0.2 * torch.rand(3, 3),
        torch.rand(3, 3),
        torch.full((3,), 0.5),  # well below MAX_ALPHA, so the cap never switches on
    )
    inputs = tuple(x.to(dtype).requires_grad_() for x in inputs)
    assert torch.autograd.gradcheck(lambda *xs: render(*xs, cam), inputs)


def ring(n, radius, height, offset, size):
    """n cameras on a horizontal circle, all looking at the origin."""
    angles = [2 * math.pi * k / n + offset for k in range(n)]
    eyes = [(radius * math.cos(a), radius * math.sin(a), height) for a in angles]
    return [look_at(e, (0.0, 0.0, 0.0), size, size) for e in eyes]


def make_shape(shape, n):
    """Blob centres (n, 3) and colours (n, 3) in [0, 1] for the true scene."""
    if shape == "torus":
        u = torch.rand(n) * 2 * math.pi  # around the ring
        v = torch.rand(n) * 2 * math.pi  # around the tube
        ring_r, tube_r = 0.8, 0.15  # a thin tube, so the hole stays visible
        means = torch.stack(
            [
                (ring_r + tube_r * torch.cos(v)) * torch.cos(u),
                (ring_r + tube_r * torch.cos(v)) * torch.sin(u),
                tube_r * torch.sin(v),
            ],
            dim=-1,
        )
        tilt = math.radians(15)  # a slight tilt, so the hole looks different from each camera
        rot = torch.tensor(
            [[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]
        )
        means = means @ rot.T
        hue = u / (2 * math.pi)
    elif shape == "helix":
        t = torch.rand(n)
        angle = t * 4 * math.pi  # two turns
        means = torch.stack([0.6 * torch.cos(angle), 0.6 * torch.sin(angle), 1.6 * t - 0.8], dim=-1)
        means = means + 0.05 * torch.randn_like(means)  # a thick tube, not a thin line
        hue = 0.8 * t
    else:  # random
        means = torch.rand(n, 3) * 2 - 1
        hue = torch.rand(n)
    hsv = torch.stack([hue, torch.full_like(hue, 0.85), torch.full_like(hue, 0.95)], dim=-1)
    return means, torch.from_numpy(hsv_to_rgb(hsv.numpy())).float()


def show(ax, image, **kwargs):
    ax.imshow(image.detach().cpu().numpy(), interpolation="nearest", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_blobs(ax, model, title):
    means, _, scales, colors, opacities = (v.detach().cpu().numpy() for v in model.values())
    size = 4000 * scales.mean(axis=1) ** 2 * opacities  # bigger / more opaque blobs look bigger
    ax.scatter(*means.T, c=colors.clip(0, 1), s=size, alpha=0.8, linewidths=0)
    ax.set_title(title)
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60)


parser = argparse.ArgumentParser()
parser.add_argument("--shape", choices=["torus", "helix", "random"], default="torus")
parser.add_argument("--size", type=int, default=64, help="image side in pixels")
parser.add_argument("--n", type=int, default=300, help="number of blobs")
parser.add_argument("--steps", type=int, default=3000, help="number of optimization steps")
args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)
check_projection()
check_cameras()
check_gradients()
print("checks passed: projection, cameras, gradients")

device = get_device()

# Two rings of training cameras, one looking down (~37 deg) and one looking up (~27 deg),
# plus 2 held-out views at a height and angle no training camera has.
train_cams = ring(6, 4.0, 3.0, 0.0, args.size) + ring(6, 4.0, -2.0, math.pi / 6, args.size)
test_cams = ring(2, 4.0, 1.0, math.pi / 4, args.size)
train_cams = [c.to(device) for c in train_cams]
test_cams = [c.to(device) for c in test_cams]
all_cams = train_cams + test_cams

# The true scene: blobs placed on the chosen shape, with fixed parameters.
means, colors = make_shape(args.shape, args.n)
truth = Gaussians3D(args.n)
with torch.no_grad():
    truth.means.copy_(means)
    truth.color_raw.copy_(torch.logit(colors.clamp(0.02, 0.98)))
    if args.shape == "random":
        truth.log_scales.uniform_(math.log(0.04), math.log(0.12))
        truth.opacity_raw.uniform_(0.5, 3)
    else:
        truth.log_scales.uniform_(math.log(0.03), math.log(0.06))
        truth.opacity_raw.uniform_(1, 3)
truth = truth.to(device)
with torch.no_grad():
    targets = [truth(c)[0] for c in all_cams]
train_targets, test_targets = targets[: len(train_cams)], targets[len(train_cams) :]

# The start: the true scene with every parameter scrambled.
model = Gaussians3D(args.n).to(device)
with torch.no_grad():
    for p, q in zip(model.parameters(), truth.parameters(), strict=True):
        p.copy_(q + 0.3 * torch.randn_like(q) * (q.std() + 0.1))
    model.quats.normal_()  # rotations start fully random
    start = [model(c)[0].clone() for c in all_cams]
start_model = Gaussians3D(args.n).to(device)
start_model.load_state_dict(model.state_dict())


def shape_error(m):
    """Mean distance from each visible blob's centre to the nearest true blob centre.

    Blob i is not compared with true blob i: blobs can swap roles and still give the same
    images, so only "do the blobs lie on the true shape" is meaningful. Nearly transparent
    blobs (opacity < 0.1) are ignored, because the images say nothing about where they are.
    """
    visible = torch.sigmoid(m.opacity_raw) > 0.1
    return torch.cdist(m.means[visible], truth.means).min(dim=1).values.mean().item()


history = fit(model, train_cams, train_targets, steps=args.steps)

with torch.no_grad():
    final = [model(c) for c in all_cams]
start_scores = [psnr(s, t) for s, t in zip(start, targets, strict=True)]
final_scores = [psnr(f[0], t) for f, t in zip(final, targets, strict=True)]
n_train = len(train_cams)
for name, sl in [("train", slice(0, n_train)), ("held-out", slice(n_train, None))]:
    print(
        f"{name:8s} PSNR  start {np.mean(start_scores[sl]):5.1f} dB"
        f"  ->  fitted {np.mean(final_scores[sl]):5.1f} dB"
    )
print(
    f"shape error, mean distance to nearest true blob (scene is ~2 wide): "
    f"start {shape_error(start_model):.3f}  ->  fitted {shape_error(model):.3f}"
)

out = Path("outputs/synthetic3d")
out.mkdir(parents=True, exist_ok=True)

# 1. The scene itself, in 3D.
fig = plt.figure(figsize=(15, 5))
for k, (m, title) in enumerate(
    [
        (truth, f"true scene ({args.shape}, {args.n} blobs)"),
        (start_model, f"start: scrambled\nshape error {shape_error(start_model):.3f}"),
        (model, f"fitted after {args.steps} steps\nshape error {shape_error(model):.3f}"),
    ]
):
    scatter_blobs(fig.add_subplot(1, 3, k + 1, projection="3d"), m, title)
fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.88, wspace=0.05)
fig.savefig(out / "scene_3d.png", dpi=120)
plt.close(fig)

# 2. Rendered views: 4 training cameras (2 from each ring) and both held-out cameras.
cols = [0, 3, 6, 9, n_train, n_train + 1]
row_names = ["target", "start", "fitted", "|fitted - target|", "fitted depth\n(yellow = near)"]
fig, axes = plt.subplots(len(row_names), len(cols), figsize=(2.2 * len(cols), 2.3 * len(row_names)))
for j, i in enumerate(cols):
    image, depth, alpha = final[i]
    kind = "held-out" if i >= n_train else "train"
    axes[0, j].set_title(f"{kind} cam {i}", color="tab:red" if i >= n_train else "black")
    show(axes[0, j], targets[i].clamp(0, 1))
    show(axes[1, j], start[i].clamp(0, 1))
    axes[1, j].set_xlabel(f"{start_scores[i]:.1f} dB")
    show(axes[2, j], image.clamp(0, 1))
    axes[2, j].set_xlabel(f"{final_scores[i]:.1f} dB")
    show(axes[3, j], (image - targets[i]).abs().mean(-1), cmap="magma", vmin=0, vmax=0.3)
    # Depth of covered pixels (alpha-weighted depth / alpha); near = bright.
    d = torch.where(alpha > 0.5, depth / alpha.clamp(min=1e-6), torch.nan)
    show(axes[4, j], d, cmap="viridis_r")
for r, name in enumerate(row_names):
    axes[r, 0].set_ylabel(name, fontsize=11)
fig.suptitle("PSNR vs target under each image; held-out cameras (red) were never trained on")
fig.tight_layout()
fig.savefig(out / "views.png", dpi=120)
plt.close(fig)

# 3. Training curve.
fig, ax = plt.subplots(figsize=(6, 3.5))
steps, scores = zip(*history, strict=True)
ax.plot(steps, scores)
ax.axhline(np.mean(final_scores[n_train:]), color="tab:red", ls="--", label="held-out (final)")
ax.set_xlabel("step")
ax.set_ylabel("mean train PSNR (dB)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "training.png", dpi=120)
plt.close(fig)

# 4. The held-out cameras alone, big. Same positions as test_cams, but 3x the pixels and a
# zoom lens (35 deg instead of 60 deg) so the shape fills the frame. The fitted blobs are a
# continuous 3D scene, so they can be rendered at any resolution and zoom.
big = 3 * args.size
big_cams = [
    look_at(c.R.T @ -c.t, (0.0, 0.0, 0.0), big, big, fov_deg=35.0).to(device)  # eye = -R^T t
    for c in (c.to("cpu") for c in test_cams)
]
with torch.no_grad():
    big_true = [truth(c)[0] for c in big_cams]
    big_fit = [model(c)[0] for c in big_cams]
fig, axes = plt.subplots(2, 2, figsize=(10, 10.6))
for r, (t, f) in enumerate(zip(big_true, big_fit, strict=True)):
    show(axes[r, 0], t.clamp(0, 1))
    show(axes[r, 1], f.clamp(0, 1))
    axes[r, 0].set_ylabel(f"held-out cam {n_train + r}", fontsize=12)
    axes[r, 1].set_xlabel(f"{psnr(f, t):.1f} dB vs true", fontsize=11)
axes[0, 0].set_title("true scene", fontsize=13)
axes[0, 1].set_title("fitted (never trained on this view)", fontsize=13)
fig.suptitle(
    f"Held-out cameras, zoomed in, at {big}x{big} px (training images were "
    f"{args.size}x{args.size}, so finer detail here was never seen)"
)
fig.tight_layout()
fig.savefig(out / "heldout.png", dpi=120)
plt.close(fig)

print(f"saved scene_3d.png, views.png, training.png, heldout.png to {out}")
