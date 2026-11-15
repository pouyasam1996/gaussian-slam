"""Step 4, Stage C: correct the camera poses while refining the Gaussians.

Each training camera gets a correction: a small rotation (3 numbers, axis * angle in
radians) and a shift (3 numbers, metres), both starting at 0, applied like noise.py:
  R' = dR @ R,   t' = dR @ t + dt
The renderer is plain PyTorch, so the image error flows back to these 6 numbers.

Loss = L1(colour) + depth_weight * L1(depth)            (depth only where the sensor measured)
       + anchor_weight * mean(excess^2)                  (Gaussian steps only)
  excess = how far a Gaussian has moved beyond `radius` from where fusion put it. Without
  it, Gaussians "cheat": they move to fit the wrong camera instead of the camera moving.

Three schedules, each with the same budget of rendered images (`steps`):
  alternating  Gaussian block: `gauss_block` steps on random cameras, cameras frozen.
               Camera pass: every camera in turn, Gaussians frozen, up to `track_iters`
               steps each, stopping early once its correction moves less than `stop`.
               Repeat. (Like tracking / mapping in Gaussian SLAM.)
  joint        every step updates both the Gaussians and that step's camera, after
               `warmup` Gaussian-only steps. (Like InstantSplat.)
  none         Gaussians only; the cameras stay as given. The "do nothing" reference.

Camera learning rates start large and shrink by a constant factor each step, to
`lr_pose_final` of their start value at the last step: big steps early cover the error
quickly, small steps late stop the cameras jittering around the answer.
"""

import math
import random
import time
from dataclasses import replace

import torch
from torch import nn

from gaussian_slam.geometry import Camera, quat_to_rotmat
from gaussian_slam.metrics import psnr


class PoseCorrection(nn.Module):
    """One small parameter pair per camera, so each camera can have its own optimizer.

    With one (n, 3) tensor and one Adam, cameras that were not rendered get a gradient of 0,
    but Adam's momentum keeps moving them in their last direction for ~10-20 steps.
    """

    def __init__(self, n: int):
        super().__init__()
        self.rot = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(n)])  # radians
        self.trans = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(n)])  # metres

    def optimizers(self, lr_rot: float, lr_trans: float) -> list[torch.optim.Adam]:
        """One Adam per camera; step only the one of the camera that was rendered."""
        return [
            torch.optim.Adam([{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_trans}])
            for r, t in zip(self.rot, self.trans, strict=True)
        ]

    def forward(self, cam: Camera, i: int) -> Camera:
        """Camera i with its correction applied (differentiable)."""
        v = self.rot[i]
        # sqrt(... + tiny) keeps the gradient finite at angle 0, where every correction starts
        angle = torch.sqrt((v * v).sum() + 1e-12)
        q = torch.cat([torch.cos(angle / 2)[None], torch.sin(angle / 2) / angle * v])
        dR = quat_to_rotmat(q[None])[0]
        return replace(cam, R=dR @ cam.R, t=dR @ cam.t + self.trans[i])


def rotation_error_deg(c: Camera, g: Camera) -> float:
    """Angle of the rotation between two camera orientations, in degrees."""
    cos = ((c.R @ g.R.T).trace() - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, cos.item()))))


def pose_errors(cams: list[Camera], truth: list[Camera]):
    """Mean rotation error (degrees) and camera position error (cm) against the true poses."""
    rot, pos = [], []
    for c, g in zip(cams, truth, strict=True):
        rot.append(rotation_error_deg(c, g))
        centre_c, centre_g = -c.R.T @ c.t, -g.R.T @ g.t  # camera position in the world
        pos.append((centre_c - centre_g).norm().item() * 100)
    return sum(rot) / len(rot), sum(pos) / len(pos)


def trajectory_report(cams: list[Camera], truth: list[Camera]) -> dict:
    """Camera position and rotation errors against the true poses, the standard way.

    raw:     errors as they are
    aligned: first the best single rotation + shift of the whole estimated trajectory onto
             the true one (least squares), then the errors. This removes a shift of the whole
             scene, which no method can see from images alone, so only the shape of the
             trajectory is judged ("absolute trajectory error", ATE).
    RMSE = sqrt(mean(error^2)); it punishes a few large errors more than the mean does.
    """
    est = torch.stack([(-c.R.T @ c.t).cpu().double() for c in cams])  # camera centres
    gt = torch.stack([(-g.R.T @ g.t).cpu().double() for g in truth])
    rot = torch.tensor([rotation_error_deg(c, g) for c, g in zip(cams, truth, strict=True)])

    # Best rotation R and shift t with R @ est + t ~ gt (closed form, via SVD).
    me, mg = est.mean(0), gt.mean(0)
    U, _, Vt = torch.linalg.svd((est - me).T @ (gt - mg))
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))  # avoid a mirror image
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d.item()], dtype=est.dtype)) @ U.T
    aligned = (est - me) @ R.T + mg

    raw = (est - gt).norm(dim=-1) * 100  # cm
    ate = (aligned - gt).norm(dim=-1) * 100
    return {
        "pos_raw_rmse_cm": raw.pow(2).mean().sqrt().item(),
        "pos_raw_median_cm": raw.median().item(),
        "pos_raw_max_cm": raw.max().item(),
        "ate_rmse_cm": ate.pow(2).mean().sqrt().item(),
        "ate_median_cm": ate.median().item(),
        "ate_max_cm": ate.max().item(),
        "rot_rmse_deg": rot.pow(2).mean().sqrt().item(),
        "rot_median_deg": rot.median().item(),
        "rot_max_deg": rot.max().item(),
        "per_camera_cm": raw.tolist(),
        "per_camera_deg": rot.tolist(),
    }


def refine(
    model,
    poses: PoseCorrection,
    cams: list[Camera],
    images: list[torch.Tensor],
    depths: list[torch.Tensor],
    mode: str = "alternating",
    steps: int = 4000,
    gauss_block: int = 300,
    track_iters: int = 5,
    stop: float = 1e-4,
    warmup: int = 300,
    depth_weight: float = 1.0,
    anchor_weight: float = 100.0,
    radius: float = 0.05,
    lr_means: float = 2e-4,
    lr_colors: float = 0.0025,
    lr_opacity: float = 0.01,
    lr_scales: float = 0.002,
    lr_quats: float = 0.001,
    lr_rot: float = 0.005,
    lr_trans: float = 0.005,
    lr_pose_final: float = 0.1,
    background=None,
    tile_options: dict | None = None,
    truth: list[Camera] | None = None,
    log_every: int = 200,
    eval_views: int = 5,
):
    """Returns [(step, seconds, mean PSNR, rotation error deg, position error cm), ...].

    lr_rot (radians) and lr_trans (metres): camera learning rates at the start; they decay
    exponentially to lr_pose_final times that at the last step.
    """
    if mode not in ("alternating", "joint", "none"):
        raise ValueError(f"unknown mode {mode!r}")
    tile_options = tile_options or {}
    n = len(cams)
    eval_ids = [round(k * (n - 1) / max(1, eval_views - 1)) for k in range(eval_views)]
    anchor = model.means.detach().clone()
    gauss_params = [model.means, model.color_raw, model.opacity_raw, model.log_scales, model.quats]
    g_opt = torch.optim.Adam(
        [
            {"params": [model.means], "lr": lr_means},
            {"params": [model.color_raw], "lr": lr_colors},
            {"params": [model.opacity_raw], "lr": lr_opacity},
            {"params": [model.log_scales], "lr": lr_scales},
            {"params": [model.quats], "lr": lr_quats},
        ]
    )
    p_opts = poses.optimizers(lr_rot, lr_trans)  # one per camera
    history = []
    step = 0
    t0 = time.time()

    def log():
        with torch.no_grad():
            corrected = [poses(c, k) for k, c in enumerate(cams)]
            score = sum(psnr(model(corrected[k], background, **tile_options)[0], images[k])
                        for k in eval_ids) / len(eval_ids)  # fmt: skip
            rot, pos = pose_errors(corrected, truth) if truth else (math.nan, math.nan)
        history.append((step, time.time() - t0, score, rot, pos))
        print(
            f"{mode:11s} step {step:5d}  PSNR {score:.2f} dB  pose error {rot:.3f} deg {pos:.2f} cm"
        )

    def one_step(i: int, train_gauss: bool, train_pose: bool):
        """One rendered image of camera i; updates what is switched on."""
        nonlocal step
        for p in gauss_params:
            p.requires_grad_(train_gauss)
        poses.requires_grad_(train_pose)
        image, depth, alpha, _ = model(poses(cams[i], i), background, **tile_options)
        loss = torch.mean(torch.abs(image - images[i]))
        if depth_weight > 0:
            valid = (depths[i] > 0) & (alpha > 0.5)
            if valid.any():
                rendered = depth / alpha.clamp(min=1e-6)  # renderer gives alpha-weighted depth
                loss = loss + depth_weight * torch.abs(rendered - depths[i])[valid].mean()
        if train_gauss and anchor_weight > 0:
            excess = ((model.means - anchor).norm(dim=-1) - radius).clamp(min=0)
            loss = loss + anchor_weight * (excess**2).mean()
        g_opt.zero_grad()
        p_opts[i].zero_grad()
        loss.backward()
        if train_gauss:
            g_opt.step()
        if train_pose:
            # Exponential decay by overall progress, so every camera follows the same schedule
            # however often it is picked.
            factor = lr_pose_final ** (step / max(1, steps - 1))
            rot_group, trans_group = p_opts[i].param_groups
            rot_group["lr"], trans_group["lr"] = lr_rot * factor, lr_trans * factor
            p_opts[i].step()  # only camera i: the others are untouched, momentum included
        step += 1
        if step % log_every == 0:
            log()

    log()  # step 0: before any update
    if mode == "alternating":
        while step < steps:
            for _ in range(min(gauss_block, steps - step)):
                one_step(random.randrange(n), True, False)
            for i in range(n):  # camera pass, Gaussians frozen
                for _ in range(track_iters):
                    if step >= steps:
                        break
                    before = torch.cat([poses.rot[i], poses.trans[i]]).detach().clone()
                    one_step(i, False, True)
                    after = torch.cat([poses.rot[i], poses.trans[i]]).detach()
                    if (after - before).abs().max() < stop:
                        break
    else:
        while step < steps:
            train_pose = mode == "joint" and step >= warmup
            one_step(random.randrange(n), True, train_pose)
    if step % log_every:
        log()
    for p in gauss_params:
        p.requires_grad_(True)
    return history
