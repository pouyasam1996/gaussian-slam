"""Step 4, Stage D test: re-fuse with the corrected poses, rebuild the Gaussians, refine again.

uv run python scripts/step4_fusion/refuse_rounds.py                          # 3 rounds x 1000 steps
uv run python scripts/step4_fusion/refuse_rounds.py --rounds 6 --steps 500   # same budget, 6 rounds

One round:
  1. Fuse all training depth images with the current poses (a new grid every round).
  2. Snap every camera onto that surface with its own depth image (align.py, exact 6x6
     solves, no rendering), then fuse again with the snapped poses. --icp-iters 0 = off.
  3. Build Gaussians from that surface (Stage B), new pose corrections at 0.
  4. Refine both with the joint schedule (Stage C) for --steps rendered images.
     --steps 0 = no Gaussians: fusion + snapping only (the classical loop, very fast).
  5. The corrected poses become the starting poses of the next round.
With --icp-iters 0, round 1 starts from the noisy poses, so 1 round x 3000 steps is exactly
refine_poses.py with --modes joint --steps 3000: the baseline to compare against.

Stops early when, between two rounds, the cameras moved less than --stop-deg and --stop-mm on
average (the poses stopped changing).

Besides pose errors, every round measures how sharp the fused surface is: the distance from
each surface point to the surface fused with the true poses (needs Open3D). Wrong poses make
walls thick and doubled, so this distance falls as the poses improve.

Outputs in outputs/replica_<scene>/:
  stage_d_curves.png        pose errors and PSNR against seconds (fusion included), rounds marked
  stage_d_surface_<r>.ply   fused surface points of each round (round 0 = true poses)
  stage_d_rounds.csv        one row per round
  stage_d_views.png         3 training frames: target / start / each round (each drawn from
                            its own corrected cameras); not with --steps 0
  stage_d_trajectory.png    camera positions from above: true path, and each camera's error
                            drawn 20x longer, for the start and each round
  stage_d_gaussians_<n>.ply blob centres with colour: start, round1, round2, ...
  stage_d_round<r>.pt       each round's refined model and its corrected cameras
"""

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from gaussian_slam.device import get_device
from gaussian_slam.metrics import psnr
from gaussian_slam.step4_fusion.align import snap_cameras
from gaussian_slam.step4_fusion.init import from_surface
from gaussian_slam.step4_fusion.noise import PoseNoise
from gaussian_slam.step4_fusion.refine import (
    PoseCorrection,
    refine,
    rotation_error_deg,
    trajectory_report,
)
from gaussian_slam.step4_fusion.replica import frames
from gaussian_slam.step4_fusion.report import (
    save_checkpoint,
    save_gaussians,
    save_trajectory,
    save_views,
)
from gaussian_slam.step4_fusion.surface import extract_points, save_ply
from gaussian_slam.step4_fusion.tsdf import TSDFVolume

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Replica/room0", help="scene folder")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--steps", type=int, default=1000, help="rendered images per round")
parser.add_argument("--every", type=int, default=20, help="training frame spacing")
parser.add_argument("--stride", type=int, default=4, help="training image: every n-th pixel")
parser.add_argument("--fuse-stride", type=int, default=2, help="depth used for fusion")
parser.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
parser.add_argument("--min-weight", type=float, default=1.0)
parser.add_argument("--noise", choices=["random", "drift"], default="random")
parser.add_argument("--rot", type=float, default=1.0, help="pose noise, degrees per frame")
parser.add_argument("--trans", type=float, default=0.02, help="pose noise, metres per frame")
parser.add_argument("--warmup", type=int, default=300, help="Gaussian-only steps each round")
parser.add_argument("--depth-weight", type=float, default=1.0)
parser.add_argument("--anchor-weight", type=float, default=100.0)
parser.add_argument("--stop-deg", type=float, default=0.01, help="stop: mean rotation change")
parser.add_argument("--stop-mm", type=float, default=0.5, help="stop: mean position change")
parser.add_argument("--budget-mb", type=float, default=512)
parser.add_argument("--icp-iters", type=int, default=5, help="depth snapping per round, 0 = off")
parser.add_argument("--icp-max-dist", type=float, default=0.1, help="ignore pairs further (m)")
parser.add_argument("--icp-sample", type=int, default=4, help="every n-th depth pixel")
parser.add_argument(
    "--no-refuse-after-snap", action="store_true", help="keep the surface fused before snapping"
)
args = parser.parse_args()

device = get_device()
out = Path("outputs") / f"replica_{Path(args.data).name}"
out.mkdir(parents=True, exist_ok=True)
black = torch.zeros(3, device=device)
tile_options = {"budget_mb": args.budget_mb}

# 1. Data, loaded once. Training images (small) and fusion depth/colour (larger) of the same
# frames; held-out frames halfway between, scored with their true poses.
train = [(cam, depth, color) for _, cam, depth, color in frames(args.data, args.every, args.stride)]
fuse_data = [(cam, depth, color) for _, cam, depth, color in frames(args.data, args.every, args.fuse_stride)]  # fmt: skip
test = [
    (c, im) for i, c, _, im in frames(args.data, args.every // 2, args.stride) if i % args.every
]
truth = [c for c, _, _ in train]
noise = PoseNoise(args.noise, args.rot, args.trans, seed=0)
noisy = [noise(c) for c in truth]

images = [im.to(device) for _, _, im in train]
depths = [d.to(device) for _, d, _ in train]
truth_dev = [c.to(device) for c in truth]
test_cams = [c.to(device) for c, _ in test]
test_images = [im.to(device) for _, im in test]


def fuse(poses):
    """New TSDF from all fusion frames with these poses (R, t taken from `poses`)."""
    vol = TSDFVolume(args.voxel)
    for (cam, depth, color), p in zip(fuse_data, poses, strict=True):
        vol.integrate(depth, color, replace(cam, R=p.R.cpu(), t=p.t.cpu()))
    return extract_points(vol, args.min_weight)


def to_cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.double().cpu().numpy()))


def surface_error_mm(points, reference) -> float:
    """Mean distance from each surface point to the surface fused with the true poses."""
    return float(np.asarray(to_cloud(points).compute_point_cloud_distance(reference)).mean() * 1000)


# Reference surface: fused with the true poses (what perfect cameras would give).
ref_points, ref_colors, ref_normals = fuse(truth)
reference = to_cloud(ref_points)
save_ply(out / "stage_d_surface_0.ply", ref_points, ref_colors, ref_normals)

# 2. Rounds.
cams = [c.to(device) for c in noisy]  # starting poses of the current round
start = trajectory_report(cams, truth_dev)
print(
    f"{len(cams)} training frames; start: ATE {start['ate_rmse_cm']:.2f} cm, "
    f"rotation {start['rot_rmse_deg']:.3f} deg (RMSE)"
)
noisy_dev = cams
models, used_cams, reports = {}, {"start": cams}, {"start": start}  # for the pictures
frame_ids = [k * args.every for k in range(len(cams))]
rows, timeline = [], []  # timeline: (seconds, rotation deg, position cm, PSNR, round)
clock = 0.0  # seconds since the start, fusion included
for r in range(1, args.rounds + 1):
    print(f"\n=== round {r} ===")
    round_start = cams
    t0 = time.time()
    points, colors, normals = fuse(cams)
    fuse_time = time.time() - t0
    clock += fuse_time
    surf = surface_error_mm(points, reference)
    print(f"fusion {fuse_time:.1f} s, {len(points)} points, surface error {surf:.1f} mm")
    if r == 1 and args.steps > 0:  # the start: Gaussians from the noisy poses, before snapping
        torch.manual_seed(0)
        random.seed(0)
        models["start"] = from_surface(points, colors, normals, args.voxel).to(device)

    # Snap every camera onto the fused surface with its depth image, then fuse again with
    # the snapped poses so the Gaussians start from the sharper surface.
    if args.icp_iters > 0:
        t0 = time.time()
        cams, gap_before, gap_after = snap_cameras(
            cams, depths, points.to(device), normals.to(device),
            iters=args.icp_iters, max_dist=args.icp_max_dist, sample=args.icp_sample,
        )  # fmt: skip
        snap = trajectory_report(cams, truth_dev)
        clock += time.time() - t0
        print(
            f"snap: {time.time() - t0:.1f} s, mean gap {gap_before:.1f} -> {gap_after:.1f} mm; "
            f"ATE {snap['ate_rmse_cm']:.2f} cm, rotation {snap['rot_rmse_deg']:.3f} deg"
        )
        timeline.append((clock, snap["rot_rmse_deg"], snap["ate_rmse_cm"], float("nan"), r))
        if not args.no_refuse_after_snap:
            t0 = time.time()
            points, colors, normals = fuse(cams)
            clock += time.time() - t0
            surf = surface_error_mm(points, reference)
            print(f"fusion after snap: {len(points)} points, surface error {surf:.1f} mm")
    save_ply(out / f"stage_d_surface_{r}.ply", points, colors, normals)

    history = [(0, 0.0, float("nan"), float("nan"), float("nan"))]
    held = float("nan")
    new = cams
    if args.steps > 0:  # --steps 0: snapping and fusion only, no Gaussians
        torch.manual_seed(r)
        random.seed(r)
        model = from_surface(points, colors, normals, args.voxel).to(device)
        corrections = PoseCorrection(len(cams)).to(device)
        history = refine(
            model, corrections, cams, images, depths, mode="joint", steps=args.steps,
            warmup=args.warmup, depth_weight=args.depth_weight, anchor_weight=args.anchor_weight,
            radius=args.voxel, background=black, tile_options=tile_options, truth=truth_dev,
        )  # fmt: skip
        timeline += [(clock + s, rot, pos, p, r) for _, s, p, rot, pos in history]
        clock += history[-1][1]
        with torch.no_grad():
            new = [corrections(c, k) for k, c in enumerate(cams)]
            new = [replace(c, R=c.R.detach().clone(), t=c.t.detach().clone()) for c in new]
            held = sum(psnr(model(c, black, **tile_options)[0], im)
                       for c, im in zip(test_cams, test_images, strict=True)) / len(test_cams)  # fmt: skip
    moved_deg = sum(rotation_error_deg(a, b) for a, b in zip(new, round_start, strict=True)) / len(
        new
    )
    moved_mm = (
        sum(
            ((-a.R.T @ a.t) - (-b.R.T @ b.t)).norm().item()
            for a, b in zip(new, round_start, strict=True)
        )
        / len(new)
        * 1000
    )
    cams = new
    rep = trajectory_report(cams, truth_dev)
    name = f"round {r}"
    used_cams[name], reports[name] = cams, rep
    if args.steps > 0:
        models[name] = model
        save_checkpoint(
            out / f"stage_d_round{r}.pt", model, cams, truth_dev, noisy_dev, frame_ids, vars(args)
        )
    rows.append(
        (r, fuse_time, len(points), surf, rep["ate_rmse_cm"], rep["ate_median_cm"],
         rep["rot_rmse_deg"], rep["rot_median_deg"], history[-1][2], held, moved_deg, moved_mm,
         clock)
    )  # fmt: skip
    print(
        f"round {r}: ATE {rep['ate_rmse_cm']:.2f} cm, rotation {rep['rot_rmse_deg']:.3f} deg, "
        f"held-out {held:.2f} dB; cameras moved {moved_deg:.3f} deg, {moved_mm:.2f} mm"
    )
    if moved_deg < args.stop_deg and moved_mm < args.stop_mm:
        print("poses stopped changing: done")
        break

# The surface the final poses give (what the next round would start from).
points, colors, normals = fuse(cams)
final_surf = surface_error_mm(points, reference)
save_ply(out / f"stage_d_surface_{len(rows) + 1}.ply", points, colors, normals)

# 3. Report.
print(
    f"\nstart: ATE {start['ate_rmse_cm']:.2f} / {start['ate_median_cm']:.2f} cm, "
    f"rotation {start['rot_rmse_deg']:.3f} / {start['rot_median_deg']:.3f} deg (RMSE / median)"
)
print(
    "round  surface   ATE RMSE / median     rotation RMSE / median   train   held-out   moved            time"
)
for r, _fuse_s, _points, surf, ate, ate_med, rot, rot_med, score, held, mdeg, mmm, t in rows:
    print(
        f"{r:5d} {surf:6.1f} mm  {ate:6.2f} / {ate_med:5.2f} cm   {rot:6.3f} / {rot_med:5.3f} deg"
        f"   {score:5.2f}   {held:5.2f} dB   {mdeg:.3f} deg {mmm:5.2f} mm   {t:7.1f} s"
    )
print(f"surface with the final poses: {final_surf:.1f} mm from the true-pose surface")

with (out / "stage_d_rounds.csv").open("w") as f:
    f.write(
        "round,fusion_s,points,surface_mm,ate_rmse_cm,ate_median_cm,rot_rmse_deg,rot_median_deg,"
        "train_psnr,heldout_psnr,moved_deg,moved_mm,seconds\n"
    )
    for row in rows:
        f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
t = [x[0] for x in timeline]
for ax, col, label in zip(
    axes,
    (1, 2, 3),
    ("rotation error (deg)", "position error (cm)", "training PSNR (dB)"),
    strict=True,
):
    ax.plot(t, [x[col] for x in timeline], ".-", ms=3)
    for row in rows[:-1]:
        ax.axvline(row[-1], color="0.7", ls="--", lw=0.8)  # end of a round = next fusion
    ax.set_xlabel("seconds (fusion included); dashed = re-fusion")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out / "stage_d_curves.png", dpi=120)
print(f"saved stage_d_curves.png, stage_d_rounds.csv, stage_d_surface_*.ply to {out}")

# 4. Pictures for a visual check (same as stage C).
save_trajectory(out / "stage_d_trajectory.png", used_cams, reports, truth_dev)
if models:  # --steps 0 has no Gaussians to draw
    save_views(
        out / "stage_d_views.png", models, used_cams, reports, images, frame_ids, black,
        tile_options,
    )  # fmt: skip
    for name, m in models.items():
        save_gaussians(out / f"stage_d_gaussians_{name.replace(' ', '')}.ply", m)
    print(f"saved stage_d_views.png, stage_d_gaussians_*.ply, stage_d_round*.pt to {out}")
print(f"saved stage_d_trajectory.png to {out}")
