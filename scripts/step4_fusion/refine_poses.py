"""Step 4, Stage C test: fix wrong camera poses while refining Gaussians, three ways.

uv run python scripts/step4_fusion/refine_poses.py                    # all three, 4000 steps each
uv run python scripts/step4_fusion/refine_poses.py --steps 1000       # quicker look
uv run python scripts/step4_fusion/refine_poses.py --modes alternating joint

1. Add pose noise to every training frame (default: random, 1 deg, 2 cm per frame).
2. Fuse with the noisy poses (the shape gets blurry, as it would with a real tracker).
3. Build the Gaussians from that surface (Stage B) and refine with each schedule:
     none         Gaussians only, cameras stay wrong (reference)
     alternating  Gaussian block, then a camera pass with Gaussians frozen, repeat
     joint        both every step, after a short Gaussian-only warm-up
   Each mode starts from the same Gaussians and gets the same number of rendered images.
4. Report pose error against the true poses, training PSNR and held-out PSNR.
   Held-out frames are rendered with their true poses: rough, because the refined model
   can sit slightly shifted from the true world frame.

5. Camera poses against ground truth, for start (noisy) and every mode: position error raw
   and after aligning the whole trajectory (ATE), and rotation error; RMSE / median / max.

Outputs in outputs/replica_<scene>/:
  stage_c_curves.png          rotation error, position error and PSNR against seconds, per mode
  stage_c_cameras.csv         per training frame: position error (cm) and rotation error (deg)
  stage_c_views.png           3 training frames: target / start / each mode (rendered with
                              that mode's own corrected camera)
  stage_c_trajectory.png      camera positions seen from above: true path, and the error of
                              each camera as a line, drawn 20x longer so it is visible
  stage_c_gaussians_<m>.ply   blob centres with colour, for the start and each mode
  stage_c_<mode>.pt           the refined model (state_dict) and the corrected cameras

View the point clouds:
  uv run python -c "import open3d as o3d; o3d.visualization.draw_geometries(
      [o3d.io.read_point_cloud('outputs/replica_room0/stage_c_gaussians_joint.ply')])"
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
from gaussian_slam.step4_fusion.init import from_surface
from gaussian_slam.step4_fusion.noise import PoseNoise
from gaussian_slam.step4_fusion.refine import (
    PoseCorrection,
    pose_errors,
    refine,
    trajectory_report,
)
from gaussian_slam.step4_fusion.replica import frames
from gaussian_slam.step4_fusion.report import (
    save_checkpoint,
    save_gaussians,
    save_trajectory,
    save_views,
)
from gaussian_slam.step4_fusion.surface import extract_points
from gaussian_slam.step4_fusion.tsdf import TSDFVolume

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Replica/room0", help="scene folder")
parser.add_argument("--modes", nargs="+", default=["none", "alternating", "joint"])
parser.add_argument("--every", type=int, default=20, help="training frame spacing")
parser.add_argument("--stride", type=int, default=4, help="training image: every n-th pixel")
parser.add_argument("--fuse-stride", type=int, default=2, help="depth used for fusion")
parser.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
parser.add_argument("--min-weight", type=float, default=1.0)
parser.add_argument("--noise", choices=["random", "drift"], default="random")
parser.add_argument("--rot", type=float, default=1.0, help="pose noise, degrees per frame")
parser.add_argument("--trans", type=float, default=0.02, help="pose noise, metres per frame")
parser.add_argument("--steps", type=int, default=4000, help="rendered images per mode")
parser.add_argument("--gauss-block", type=int, default=300, help="alternating: Gaussian steps")
parser.add_argument("--track-iters", type=int, default=5, help="alternating: steps per camera")
parser.add_argument("--warmup", type=int, default=300, help="joint: Gaussian-only steps first")
parser.add_argument(
    "--depth-weight", type=float, default=1.0, help="depth L1 (m) vs colour L1; 1 = similar pull"
)
parser.add_argument("--anchor-weight", type=float, default=100.0)
parser.add_argument("--lr-rot", type=float, default=0.005, help="camera rotation lr (radians)")
parser.add_argument("--lr-trans", type=float, default=0.005, help="camera shift lr (metres)")
parser.add_argument(
    "--lr-pose-final", type=float, default=0.1, help="camera lrs decay to this share by the end"
)
parser.add_argument("--budget-mb", type=float, default=512)
args = parser.parse_args()

device = get_device()
out = Path("outputs") / f"replica_{Path(args.data).name}"
out.mkdir(parents=True, exist_ok=True)
black = torch.zeros(3, device=device)
tile_options = {"budget_mb": args.budget_mb}

# 1. Training frames with true and noisy poses; held-out frames halfway between them.
train = [(cam, depth, color) for _, cam, depth, color in frames(args.data, args.every, args.stride)]
truth = [c for c, _, _ in train]
noise = PoseNoise(args.noise, args.rot, args.trans, seed=0)
noisy = [noise(c) for c in truth]
test = [
    (c, im) for i, c, _, im in frames(args.data, args.every // 2, args.stride) if i % args.every
]
rot0, pos0 = pose_errors(noisy, truth)
print(f"{len(train)} training frames, pose error at start: {rot0:.3f} deg, {pos0:.2f} cm")

# 2. Fuse with the noisy poses (same R, t; intrinsics of the fusion image size).
t0 = time.time()
vol = TSDFVolume(args.voxel)
for k, (_, cam, depth, color) in enumerate(frames(args.data, args.every, args.fuse_stride)):
    vol.integrate(depth, color, replace(cam, R=noisy[k].R, t=noisy[k].t))
points, colors, normals = extract_points(vol, args.min_weight)
fuse_time = time.time() - t0
print(f"fusion with noisy poses: {fuse_time:.1f} s, {len(points)} surface points")

cams = [c.to(device) for c in noisy]
images = [im.to(device) for _, _, im in train]
depths = [d.to(device) for _, d, _ in train]
truth = [c.to(device) for c in truth]
test_cams = [c.to(device) for c, _ in test]
test_images = [im.to(device) for _, im in test]

# 3. Each schedule from the same start.
results, curves = {}, {}
reports = {"start": trajectory_report(cams, truth)}  # the noisy poses, before any correction
start_model = from_surface(points, colors, normals, args.voxel).to(device)
models = {"start": start_model}  # for the pictures and point clouds at the end
corrected = {"start": cams}  # cameras used by each model
for mode in args.modes:
    print(f"\n--- {mode} ---")
    torch.manual_seed(0)
    random.seed(0)
    model = from_surface(points, colors, normals, args.voxel).to(device)
    poses = PoseCorrection(len(cams)).to(device)
    history = refine(
        model, poses, cams, images, depths, mode=mode, steps=args.steps,
        gauss_block=args.gauss_block, track_iters=args.track_iters, warmup=args.warmup,
        depth_weight=args.depth_weight, anchor_weight=args.anchor_weight, radius=args.voxel,
        lr_rot=args.lr_rot, lr_trans=args.lr_trans, lr_pose_final=args.lr_pose_final,
        background=black, tile_options=tile_options, truth=truth,
    )  # fmt: skip
    with torch.no_grad():
        held = sum(psnr(model(c, black, **tile_options)[0], im)
                   for c, im in zip(test_cams, test_images, strict=True)) / len(test_cams)  # fmt: skip
    _, seconds, score, rot, pos = history[-1]
    results[mode] = (rot, pos, score, held, seconds)
    curves[mode] = history
    with torch.no_grad():
        corrected[mode] = [poses(c, k) for k, c in enumerate(cams)]
    reports[mode] = trajectory_report(corrected[mode], truth)
    models[mode] = model
    frame_ids = [k * args.every for k in range(len(cams))]
    save_checkpoint(
        out / f"stage_c_{mode}.pt", model, corrected[mode], truth, cams, frame_ids, vars(args)
    )

# 4. Report.
print(f"\npose error at start: {rot0:.3f} deg, {pos0:.2f} cm  (fusion {fuse_time:.1f} s)")
print("mode          rotation    position    train PSNR   held-out PSNR   time")
for mode, (rot, pos, score, held, seconds) in results.items():
    print(
        f"{mode:12s} {rot:7.3f} deg {pos:7.2f} cm {score:9.2f} dB {held:11.2f} dB {seconds:7.1f} s"
    )

print("\ncamera poses against ground truth (RMSE / median / max):")
print("mode          position raw (cm)        position aligned, ATE (cm)   rotation (deg)")
for mode, r in reports.items():
    print(
        f"{mode:12s} {r['pos_raw_rmse_cm']:6.2f} {r['pos_raw_median_cm']:6.2f} {r['pos_raw_max_cm']:6.2f}"
        f"     {r['ate_rmse_cm']:6.2f} {r['ate_median_cm']:6.2f} {r['ate_max_cm']:6.2f}"
        f"         {r['rot_rmse_deg']:6.3f} {r['rot_median_deg']:6.3f} {r['rot_max_deg']:6.3f}"
    )

# Per-camera errors, one row per training frame, for your own plots.
with (out / "stage_c_cameras.csv").open("w") as f:
    f.write("frame," + ",".join(f"{m}_cm,{m}_deg" for m in reports) + "\n")
    for k in range(len(cams)):
        row = [
            f"{r['per_camera_cm'][k]:.4f},{r['per_camera_deg'][k]:.5f}" for r in reports.values()
        ]
        f.write(f"{k * args.every}," + ",".join(row) + "\n")
print(f"saved {out / 'stage_c_cameras.csv'}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for mode, history in curves.items():
    seconds = [h[1] for h in history]
    for ax, col in zip(axes, (3, 4, 2), strict=True):
        ax.plot(seconds, [h[col] for h in history], label=mode)
for ax, label in zip(
    axes, ("rotation error (deg)", "position error (cm)", "training PSNR (dB)"), strict=True
):
    ax.set_xlabel("seconds")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig(out / "stage_c_curves.png", dpi=120)
print(f"saved {out / 'stage_c_curves.png'}")

# 5. Pictures for a visual check.
frame_ids = [k * args.every for k in range(len(cams))]
save_views(
    out / "stage_c_views.png", models, corrected, reports, images, frame_ids, black, tile_options
)
save_trajectory(out / "stage_c_trajectory.png", corrected, reports, truth)
for name, m in models.items():
    save_gaussians(out / f"stage_c_gaussians_{name}.ply", m)
print(f"saved stage_c_views.png, stage_c_trajectory.png, stage_c_gaussians_*.ply to {out}")
