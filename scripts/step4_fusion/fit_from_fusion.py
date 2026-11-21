"""Step 4, Stage B: Gaussians started from the fused surface, then refined against the photos.

uv run python scripts/step4_fusion/fit_from_fusion.py                # room0, 5 cm, 2000 steps
uv run python scripts/step4_fusion/fit_from_fusion.py --steps 500    # quicker look
uv run python scripts/step4_fusion/fit_from_fusion.py --noise random # with wrong poses

1. Fuse every 20th frame into a TSDF (CPU) and extract surface points.
2. One flat coin per surface voxel, coloured from the fusion (init.from_surface).
3. Train with step 3's fit(): no densification, one colour per blob (SH degree 0).
   Rough test: small images, few steps.
4. Score on held-out frames (halfway between the training frames).

--noise random / drift: fusion and training both use the same wrong poses, as they would
with a real tracker. Held-out frames are always scored with the true poses, so the score
shows how much the pose errors damage the model.

Outputs in outputs/replica_<scene>/:
  stage_b_psnr_<noise>.png   training PSNR against seconds (fusion time included)
  stage_b_views_<noise>.png  3 held-out frames: target / before training / after training
"""

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import torch

from gaussian_slam.device import get_device
from gaussian_slam.metrics import psnr
from gaussian_slam.step3_optimizations.gaussians3d import fit
from gaussian_slam.step4_fusion.init import from_surface
from gaussian_slam.step4_fusion.noise import PoseNoise
from gaussian_slam.step4_fusion.replica import frames
from gaussian_slam.step4_fusion.surface import extract_points
from gaussian_slam.step4_fusion.tsdf import TSDFVolume

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Replica/room0", help="scene folder")
parser.add_argument("--every", type=int, default=20, help="training frame spacing")
parser.add_argument("--stride", type=int, default=4, help="training image: every n-th pixel")
parser.add_argument("--fuse-stride", type=int, default=2, help="depth used for fusion")
parser.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
parser.add_argument("--trunc", type=float, default=4.0, help="truncation, in voxels")
parser.add_argument("--min-weight", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--width", type=float, default=0.5, help="coin radius, in voxels")
parser.add_argument("--opacity", type=float, default=0.9, help="starting opacity of the coins")
parser.add_argument("--eval-views", type=int, default=5)
parser.add_argument("--budget-mb", type=float, default=512)
parser.add_argument("--noise", choices=["none", "random", "drift"], default="none")
parser.add_argument("--rot", type=float, default=None, help="pose error, degrees (per frame)")
parser.add_argument("--trans", type=float, default=None, help="pose error, metres (per frame)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = get_device()
out = Path("outputs") / f"replica_{Path(args.data).name}"
out.mkdir(parents=True, exist_ok=True)

# Same per-frame defaults as run_fusion.py; drift steps add up, so they are much smaller.
noise = None
if args.noise != "none":
    rot_trans = {"random": (1.0, 0.02), "drift": (0.05, 0.002)}[args.noise]
    rot = args.rot if args.rot is not None else rot_trans[0]
    trans = args.trans if args.trans is not None else rot_trans[1]
    noise = PoseNoise(args.noise, rot, trans, args.seed)
    print(f"pose noise: {args.noise}, {rot} deg, {trans * 100:.1f} cm per frame")

# 1. Fusion -> surface points.
t0 = time.time()
vol = TSDFVolume(args.voxel, trunc_voxels=args.trunc)
noisy_poses = []  # (R, t) per fused frame, reused for the training frames below
for _, cam, depth, color in frames(args.data, args.every, args.fuse_stride):
    if noise is not None:
        cam = noise(cam)
    noisy_poses.append((cam.R, cam.t))
    vol.integrate(depth, color, cam)
points, colors, normals = extract_points(vol, args.min_weight)
fuse_time = time.time() - t0
print(f"fusion: {fuse_time:.1f} s, {len(points)} surface points")

# Training frames 0, 20, 40, ...; held-out frames 10, 30, 50, ... (halfway between).
# Training frames are the fused frames at another size: same (possibly wrong) pose.
train = [
    (replace(c, R=R, t=t), im)
    for (_, c, _, im), (R, t) in zip(
        frames(args.data, args.every, args.stride), noisy_poses, strict=True
    )
]
test = [
    (c, im) for i, c, _, im in frames(args.data, args.every // 2, args.stride) if i % args.every
]
train_cams = [c.to(device) for c, _ in train]
train_images = [im.to(device) for _, im in train]
test_cams = [c.to(device) for c, _ in test]
test_images = [im.to(device) for _, im in test]
black = torch.zeros(3, device=device)
tile_options = {"budget_mb": args.budget_mb}
print(
    f"{len(train_cams)} training / {len(test_cams)} held-out frames at "
    f"{train_cams[0].width}x{train_cams[0].height}, device {device}"
)

# 2. The fused start.
t0 = time.time()
model = from_surface(points, colors, normals, args.voxel, args.width, opacity=args.opacity)
model.to(device)
init_time = fuse_time + time.time() - t0
print(f"{len(model.means)} blobs, ready after {init_time:.1f} s")

ids = [0, len(test_cams) // 2, len(test_cams) - 1]


def held_out_psnr():
    with torch.no_grad():
        scores = [
            psnr(model(c, black, **tile_options)[0], im)
            for c, im in zip(test_cams, test_images, strict=True)
        ]
    return sum(scores) / len(scores)


with torch.no_grad():  # snapshots before training, for the picture
    before = [model(test_cams[k], black, **tile_options)[0].cpu() for k in ids]
held_before = held_out_psnr()

# 3. Train.
torch.manual_seed(0)
random.seed(0)
t0 = time.time()
history = fit(
    model, train_cams, train_images, steps=args.steps, sh_every=0,
    background=black, tile_options=tile_options, eval_views=args.eval_views,
)  # fmt: skip
seconds = time.time() - t0
per_step = seconds / max(history[-1][0], 1)  # includes the PSNR checks; rough
curve = [(init_time + s * per_step, p) for s, p in history]
held_after = held_out_psnr()

# 4. Report.
print(
    f"\nfusion + start: {init_time:.1f} s, training: {seconds:.1f} s for {history[-1][0]} steps\n"
    f"training PSNR: {history[0][1]:.2f} dB at step 1 -> {history[-1][1]:.2f} dB\n"
    f"held-out PSNR: {held_before:.2f} dB before training -> {held_after:.2f} dB"
)

plt.figure(figsize=(6, 4))
plt.plot([t for t, _ in curve], [p for _, p in curve])
plt.xlabel("seconds (fusion time included)")
plt.ylabel("training PSNR (dB)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out / f"stage_b_psnr_{args.noise}.png", dpi=120)

fig, axes = plt.subplots(len(ids), 3, figsize=(9, 2 * len(ids)))
with torch.no_grad():
    for row, k in enumerate(ids):
        shown = [test_images[k].cpu(), before[row], model(test_cams[k], black, **tile_options)[0]]
        for col, (img, title) in enumerate(zip(shown, ["target", "before", "after"], strict=True)):
            axes[row, col].imshow(img.clamp(0, 1).cpu().numpy())
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(out / f"stage_b_views_{args.noise}.png", dpi=120)
print(
    f"saved {out / f'stage_b_psnr_{args.noise}.png'} and {out / f'stage_b_views_{args.noise}.png'}"
)
