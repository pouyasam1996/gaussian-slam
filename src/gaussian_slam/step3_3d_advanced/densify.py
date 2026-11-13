"""Step 1.7: add blobs where detail is missing, delete blobs that do nothing.

Every step, track(): how hard is each blob pulled around on the screen? This is the size of
the gradient of its 2D centre, averaged over the views it was drawn in.

Every `every` steps, densify_and_prune():
  pulled hard and small  -> clone: add a copy; the next steps pull the two apart
  pulled hard and large  -> split: replace it by 2 blobs drawn from inside it, 1.6x smaller
  nearly transparent or huge -> prune: delete it

Every `reset_every` steps, reset_opacity(): make every blob nearly transparent. Blobs that
are needed become opaque again; the rest stay faint and get pruned.

Adam keeps two running averages per blob; they are copied, zeroed or deleted with the blob.
"""

import math

import torch
from torch import nn

from gaussian_slam.geometry import quat_to_rotmat

PARAMS = ("means", "quats", "log_scales", "color_raw", "opacity_raw")


class DensityControl:
    def __init__(
        self,
        extent: float = 1.0,
        grad_threshold: float = 2e-4,
        every: int = 100,
        start: int = 300,
        stop: int | None = None,
        reset_every: int = 1000,
        max_blobs: int = 20_000,
        min_opacity: float = 0.005,
        small: float = 0.01,
        huge: float = 0.5,
        split_shrink: float = 1.6,
    ):
        """
        extent          rough half-width of the scene; "small" and "huge" are fractions of it
        grad_threshold  average screen gradient above which a blob is cloned or split. It
                        depends on the loss and the image size: tune it with the percentiles
                        printed at every densification
        every, start, stop   densify every `every` steps from `start` to `stop` (None = to the end)
        reset_every     opacity reset period, inside the same start..stop window (0 = never)
        max_blobs       the blob count never grows past this
        """
        self.extent, self.grad_threshold = extent, grad_threshold
        self.every, self.start, self.stop, self.reset_every = every, start, stop, reset_every
        self.max_blobs, self.min_opacity = max_blobs, min_opacity
        self.small, self.huge, self.split_shrink = small, huge, split_shrink
        self.grad_sum = self.count = None

    def update(self, step, model, optimizer, info):
        """Call after optimizer.step(), with the info returned by the renderer this step."""
        self.track(model, info)
        active = step >= self.start and (self.stop is None or step <= self.stop)
        if active and step % self.every == 0:
            self.densify_and_prune(model, optimizer)
        if active and self.reset_every and step % self.reset_every == 0:
            self.reset_opacity(model, optimizer)

    @torch.no_grad()
    def track(self, model, info):
        n = model.means.shape[0]
        if self.grad_sum is None or len(self.grad_sum) != n:
            self.grad_sum = torch.zeros(n, device=model.means.device, dtype=model.means.dtype)
            self.count = torch.zeros_like(self.grad_sum)
        g = info["mean2d"].grad  # (n_projected, 2), pixels
        if g is None:
            return
        seen = info["visible"]
        ids = info["ids"][seen]  # back to indices into all blobs
        self.grad_sum.index_add_(0, ids, g[seen].norm(dim=-1))
        self.count.index_add_(0, ids, torch.ones_like(self.grad_sum[ids]))

    @torch.no_grad()
    def densify_and_prune(self, model, optimizer):
        n = model.means.shape[0]
        avg = self.grad_sum / self.count.clamp(min=1)
        size = torch.exp(model.log_scales).amax(dim=1)
        opacity = torch.sigmoid(model.opacity_raw)

        prune = (opacity < self.min_opacity) | (size > self.huge * self.extent)
        grow = (avg > self.grad_threshold) & ~prune
        # Clone and split each add one blob in total. Over the cap: keep the most pulled.
        room = max(0, self.max_blobs - (n - int(prune.sum())))
        if int(grow.sum()) > room:
            candidates = torch.nonzero(grow).squeeze(1)
            grow = torch.zeros_like(grow)
            grow[candidates[avg[candidates].topk(room).indices]] = True
        clone = grow & (size <= self.small * self.extent)
        split = grow & ~clone
        seen = avg[self.count > 0].sort().values  # for the printout: blobs drawn at least once

        params = {name: getattr(model, name) for name in PARAMS}
        halves = [split_half(params, split, self.split_shrink) for _ in range(2)]
        new_rows = {
            name: torch.cat([p[clone], halves[0][name], halves[1][name]])
            for name, p in params.items()
        }
        resize(model, optimizer, keep=~(prune | split), new_rows=new_rows)
        self.grad_sum = self.count = None  # restart the statistics for the new blob set

        print(
            f"  densify: {int(clone.sum())} cloned, {int(split.sum())} split, "
            f"{int(prune.sum())} pruned -> {model.means.shape[0]} blobs | screen grad "
            f"p50 {percentile(seen, 0.5):.1e}  p90 {percentile(seen, 0.9):.1e}  "
            f"p99 {percentile(seen, 0.99):.1e}  (threshold {self.grad_threshold:.1e})"
        )

    @torch.no_grad()
    def reset_opacity(self, model, optimizer):
        """Cap every opacity at 0.01 and forget Adam's running averages for opacity."""
        model.opacity_raw.clamp_(max=math.log(0.01 / 0.99))  # sigmoid(x) = 0.01
        state = optimizer.state.get(model.opacity_raw)
        if state:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
        print("  opacity reset")


def split_half(params, split, shrink):
    """One of the two new blobs per split blob: a point drawn from the old blob's own Gaussian,
    scales divided by `shrink`, everything else copied."""
    scales = torch.exp(params["log_scales"][split])
    R = quat_to_rotmat(params["quats"][split])
    offset = (R @ (scales * torch.randn_like(scales))[..., None]).squeeze(-1)
    return {
        "means": params["means"][split] + offset,
        "quats": params["quats"][split],
        "log_scales": params["log_scales"][split] - math.log(shrink),
        "color_raw": params["color_raw"][split],
        "opacity_raw": params["opacity_raw"][split],
    }


def resize(model, optimizer, keep, new_rows):
    """Keep rows `keep` of every parameter and append new_rows; Adam's state follows.

    Adam stores exp_avg and exp_avg_sq (running averages of the gradient and its square) with
    one row per blob. Kept blobs keep their rows, deleted ones lose them, new ones start at 0.
    The new nn.Parameter replaces the old one in the model and in the optimizer.
    """
    for name in PARAMS:
        old = getattr(model, name)
        new = nn.Parameter(torch.cat([old.detach()[keep], new_rows[name]]))
        state = optimizer.state.pop(old, None)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = torch.cat([state[key][keep], torch.zeros_like(new_rows[name])])
            optimizer.state[new] = state
        for group in optimizer.param_groups:
            group["params"] = [new if p is old else p for p in group["params"]]
        setattr(model, name, new)


def percentile(sorted_values, q):
    if len(sorted_values) == 0:
        return float("nan")
    return sorted_values[int(q * (len(sorted_values) - 1))].item()
