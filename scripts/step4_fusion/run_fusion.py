"""Step 4: fuse a Replica scene into a TSDF grid and save its surface as coloured points.

uv run python scripts/step4_fusion/run_fusion.py                        # room0, quick preview
uv run python scripts/step4_fusion/run_fusion.py --noise random         # same, with wrong poses
uv run python scripts/step4_fusion/run_fusion.py --noise drift --show   # open a 3D viewer at the end

Defaults are for a quick look: every 15th frame, images shrunk 8x (1200x680 -> 150x85),
5 cm voxels. For a detailed surface: --every 5 --stride 1 --voxel 0.02.

Outputs in outputs/replica_<scene>/:
  points_<noise>.ply  surface points with colour and normal (Open3D, MeshLab or CloudCompare)

Compare points_none.ply with points_random.ply / points_drift.ply: with wrong poses, walls
get thick and blurry and edges appear twice. That is what the next stage has to fix.
"""

import argparse
import time
from pathlib import Path

from gaussian_slam.step4_fusion.noise import PoseNoise
from gaussian_slam.step4_fusion.replica import frames
from gaussian_slam.step4_fusion.surface import extract_points, save_ply
from gaussian_slam.step4_fusion.tsdf import TSDFVolume

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Replica/room0", help="scene folder")
parser.add_argument("--every", type=int, default=15, help="use every n-th frame")
parser.add_argument("--limit", type=int, default=None, help="stop after this many frames")
parser.add_argument("--stride", type=int, default=8, help="keep every n-th pixel (8: 150x85 px)")
parser.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
parser.add_argument("--trunc", type=float, default=4.0, help="truncation, in voxels")
parser.add_argument("--depth-max", type=float, default=10.0, help="ignore depth beyond (m)")
parser.add_argument("--min-weight", type=float, default=3.0, help="voxels seen by more frames")
parser.add_argument("--noise", choices=["none", "random", "drift"], default="none")
parser.add_argument("--rot", type=float, default=None, help="pose error, degrees (per frame)")
parser.add_argument("--trans", type=float, default=None, help="pose error, metres (per frame)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cpu", help="cpu or mps")
parser.add_argument("--show", action="store_true", help="open an Open3D viewer at the end")
args = parser.parse_args()

# Defaults per noise type: drift steps add up, so they are much smaller.
defaults = {"random": (1.0, 0.02), "drift": (0.05, 0.002)}
noise = None
if args.noise != "none":
    rot = args.rot if args.rot is not None else defaults[args.noise][0]
    trans = args.trans if args.trans is not None else defaults[args.noise][1]
    noise = PoseNoise(args.noise, rot, trans, args.seed)
    print(f"pose noise: {args.noise}, {rot} deg, {trans * 100:.1f} cm per frame")

vol = TSDFVolume(args.voxel, trunc_voxels=args.trunc, depth_max=args.depth_max, device=args.device)
out = Path("outputs") / f"replica_{Path(args.data).name}"

t0 = time.time()
n = 0
for n, (i, cam, depth, color) in enumerate(
    frames(args.data, args.every, args.stride, args.limit), 1
):
    if noise is not None:
        cam = noise(cam)
    vol.integrate(depth, color, cam)
    if n % 50 == 0:
        print(
            f"frame {i:5d}: {len(vol.codes)} blocks, {vol.memory_mb():.0f} MB, "
            f"{(time.time() - t0) / n:.3f} s/frame"
        )
fuse_time = time.time() - t0

points, colors, normals = extract_points(vol, args.min_weight)
path = out / f"points_{args.noise}.ply"
save_ply(path, points, colors, normals)
print(
    f"fused {n} frames in {fuse_time:.1f} s ({fuse_time / max(n, 1):.3f} s/frame), "
    f"{len(vol.codes)} blocks, {vol.memory_mb():.0f} MB\n"
    f"{len(points)} surface points -> {path} (extraction {time.time() - t0 - fuse_time:.1f} s)"
)

if args.show:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.double().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())
    o3d.visualization.draw_geometries([pcd])
