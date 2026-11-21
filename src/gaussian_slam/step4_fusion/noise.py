"""Step 4: make camera poses wrong on purpose, to test later whether we can fix them.

Two kinds of error:
  "random"  every frame gets its own small, independent error (jitter)
  "drift"   the errors add up frame after frame, like a tracker slowly losing its way

An error is a small rotation dR and a shift dt applied in the camera's own frame:
  x_cam' = dR @ x_cam + dt   so   R' = dR @ R,  t' = dR @ t + dt
"""

import math
from dataclasses import replace

import torch

from gaussian_slam.geometry import Camera, quat_to_rotmat


def small_rotation(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation matrix for a rotation vector: direction = axis, length = angle in radians."""
    angle = rotvec.norm()
    if angle < 1e-12:
        return torch.eye(3)
    axis = rotvec / angle
    q = torch.cat([torch.cos(angle / 2)[None], torch.sin(angle / 2) * axis])
    return quat_to_rotmat(q[None])[0]


class PoseNoise:
    def __init__(self, mode: str = "random", rot_deg: float = 1.0, trans: float = 0.02, seed=0):
        """rot_deg / trans: standard deviation of the error per frame (degrees, metres).

        For "drift" these are the per-frame steps, so keep them much smaller: after n frames
        the error is about sqrt(n) times one step.
        """
        if mode not in ("random", "drift"):
            raise ValueError(f"mode must be 'random' or 'drift', got {mode!r}")
        self.mode = mode
        self.rot = math.radians(rot_deg)
        self.trans = trans
        self.gen = torch.Generator().manual_seed(seed)
        self.dR = torch.eye(3)  # error so far (drift only)
        self.dt = torch.zeros(3)

    def __call__(self, cam: Camera) -> Camera:
        dR = small_rotation(torch.randn(3, generator=self.gen) * self.rot)
        dt = torch.randn(3, generator=self.gen) * self.trans
        if self.mode == "drift":
            # New error on top of the old one: x -> dR (dR_old x + dt_old) + dt
            self.dR, self.dt = dR @ self.dR, dR @ self.dt + dt
            dR, dt = self.dR, self.dt
        dR, dt = dR.to(cam.R), dt.to(cam.t)
        return replace(cam, R=dR @ cam.R, t=dR @ cam.t + dt)
