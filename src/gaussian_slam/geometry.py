"""Step 1.3: rotations, blob shapes, cameras and projection.

Conventions used everywhere in this project:
- Quaternions are (w, x, y, z).
- Cameras follow OpenCV: x right, y down, z forward (into the scene).
- A Camera stores the world -> camera transform: x_cam = R @ x_world + t.
"""

import math
from dataclasses import dataclass, replace

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices.

    The quaternion is normalized first, so any 4 numbers give a valid rotation.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def covariance_3d(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Blob shape Sigma = R S S^T R^T from (N, 4) quaternions and (N, 3) scales -> (N, 3, 3)."""
    m = quat_to_rotmat(quats) * scales[:, None, :]  # = R @ diag(s): column j times s_j
    return m @ m.transpose(1, 2)


@dataclass
class Camera:
    R: torch.Tensor  # (3, 3) world -> camera rotation
    t: torch.Tensor  # (3,)   world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to(self, *args, **kwargs) -> "Camera":
        """Move R and t to a device and/or dtype, like Tensor.to()."""
        return replace(self, R=self.R.to(*args, **kwargs), t=self.t.to(*args, **kwargs))


def world_to_camera(points: torch.Tensor, cam: Camera) -> torch.Tensor:
    """(N, 3) world points -> (N, 3) camera-frame points."""
    return points @ cam.R.T + cam.t


def project(points_cam: torch.Tensor, cam: Camera) -> torch.Tensor:
    """Pinhole projection of (N, 3) camera-frame points -> (N, 2) pixels (u, v)."""
    x, y, z = points_cam.unbind(-1)
    return torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], dim=-1)


def look_at(
    eye, target, width: int, height: int, fov_deg: float = 60.0, up=(0.0, 0.0, 1.0)
) -> Camera:
    """OpenCV camera at ``eye`` looking at ``target``. ``fov_deg`` is the horizontal field of view.

    ``up`` is the world's up direction; it must not be parallel to the viewing direction.
    """
    eye, target, up = (torch.as_tensor(v, dtype=torch.float32) for v in (eye, target, up))
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward])  # rows = camera axes written in world coordinates
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return Camera(R, -R @ eye, f, f, width / 2, height / 2, width, height)


def opengl_to_opencv(c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blender/OpenGL camera -> world (4, 4) pose -> OpenCV world -> camera (R, t).

    OpenGL cameras look down -z with y up; OpenCV cameras look down +z with y down.
    Flipping the camera's y and z axes converts one into the other.
    """
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=c2w.dtype))
    rot_c2w = c2w[:3, :3] @ flip  # flip the camera's own axes (columns)
    R = rot_c2w.T  # inverse of a rotation is its transpose
    return R, -R @ c2w[:3, 3]
