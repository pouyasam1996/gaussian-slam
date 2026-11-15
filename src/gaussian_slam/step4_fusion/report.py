"""Step 4: pictures and files for a visual check of pose refinement (stages C and D).

Every function takes results keyed by a name ("start", "joint", "round 2", ...), in the
order they should be shown:
  models   name -> Gaussians3D
  cams     name -> list of cameras that model is drawn from (start: the noisy ones)
  reports  name -> refine.trajectory_report() of those cameras against the true ones
"""

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import torch

from gaussian_slam.metrics import psnr
from gaussian_slam.step4_fusion.surface import save_ply


def save_checkpoint(path, model, cams, truth, noisy, frame_ids, settings: dict):
    """The refined model (state_dict) with its corrected, true and noisy cameras."""

    def stack(cs):
        return torch.stack([c.R for c in cs]).cpu(), torch.stack([c.t for c in cs]).cpu()

    (R, t), (R_true, t_true), (R_noisy, t_noisy) = stack(cams), stack(truth), stack(noisy)
    torch.save(
        {
            "model": model.state_dict(),
            "frames": list(frame_ids),
            "R": R,
            "t": t,
            "R_true": R_true,
            "t_true": t_true,
            "R_noisy": R_noisy,
            "t_noisy": t_noisy,
            "args": settings,
        },
        path,
    )


@torch.no_grad()
def save_views(path, models, cams, reports, images, frame_ids, background, tile_options, n=3):
    """The n training frames with the largest starting position error: target, then each
    model drawn from its own cameras. Wrong poses show as blur and doubled edges."""
    first = next(iter(reports.values()))["per_camera_cm"]
    worst = sorted(range(len(first)), key=lambda k: -first[k])[:n]
    names = list(models)
    fig, axes = plt.subplots(n, len(names) + 1, figsize=(3 * (len(names) + 1), 2 * n))
    for row, k in enumerate(worst):
        axes[row, 0].imshow(images[k].clamp(0, 1).cpu().numpy())
        axes[row, 0].set_ylabel(f"frame {frame_ids[k]}")
        for col, name in enumerate(names, 1):
            image = models[name](cams[name][k], background, **tile_options)[0]
            axes[row, col].imshow(image.clamp(0, 1).cpu().numpy())
            axes[row, col].set_xlabel(
                f"{psnr(image, images[k]):.1f} dB, {reports[name]['per_camera_cm'][k]:.1f} cm"
            )
    for col, title in enumerate(["target", *names]):
        axes[0, col].set_title(title)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Training frames with the largest starting pose error (PSNR, camera position error)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def centres(cams) -> torch.Tensor:
    """(N, 3) camera positions in the world: -R^T t."""
    return torch.stack([(-c.R.T @ c.t).cpu() for c in cams])


def save_trajectory(path, cams, reports, truth, scale: float = 20):
    """Camera positions from above (the plane where the true path spreads the most), one
    panel per name: the true path, and each camera's error as a line `scale` times longer."""
    true_c = centres(truth)
    mid = true_c.mean(0)
    _, _, V = torch.linalg.svd(true_c - mid)
    plane = V[:2].T  # (3, 2): the two directions of largest spread
    true_2d = (true_c - mid) @ plane
    names = list(cams)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        tip = true_2d + scale * ((centres(cams[name]) - mid) @ plane - true_2d)
        ax.plot(true_2d[:, 0], true_2d[:, 1], "-", color="0.6", lw=1, label="true path")
        for a, b in zip(true_2d, tip, strict=True):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=1)
        ax.scatter(true_2d[:, 0], true_2d[:, 1], s=6, color="black", zorder=3)
        r = reports[name]
        ax.set_title(f"{name}: RMSE {r['pos_raw_rmse_cm']:.2f} cm, ATE {r['ate_rmse_cm']:.2f} cm")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0, 0].plot([], [], color="tab:red", label=f"position error x{scale:g}")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_gaussians(path, model, min_opacity: float = 0.1):
    """Blob centres with their colour as a PLY point cloud; nearly transparent blobs are
    left out, because they do not show in the images."""
    means, _, _, rgb, opacity = model.values()
    keep = opacity > min_opacity
    save_ply(path, means[keep].detach(), rgb[keep].detach())
