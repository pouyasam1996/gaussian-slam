"""Tiled renderer: each blob is evaluated only in the 16x16-pixel tiles it can reach.

Same picture as gaussians3d.render() (up to the 1/255 cut-off), with far less memory.

  1. project   blobs -> 2D ellipses (shared with render())
  2. pairs     radius where a blob fades below 1/255 -> one (blob, tile) pair for every tile
               the blob reaches, sorted by tile, then near -> far
  3. select    mode "sampled" only: in crowded tiles keep the important blobs plus a few
               random ones
  4. passes    each pass draws the next max(buckets) blobs of every tile, nearest first;
               a tile stops early once almost no light is left in it
  5. buckets   inside a pass, tiles are grouped by how many blobs they draw and padded only
               up to their bucket size; each group is sized to fit the memory budget
  6. blend     per group: the dense math of render() on (tiles, K blobs, 256 pixels),
               recomputed during backward instead of stored
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from gaussian_slam.step3_3d_advanced.gaussians3d import MAX_ALPHA, project_blobs

TILE = 16
PIX = TILE * TILE  # pixels per tile
MIN_ALPHA = 1 / 255  # weaker than this is invisible on a screen
TABLES = 12  # rough count of (tiles, K, 256) float tables alive while one group blends


def render_tiled(
    means,
    quats,
    scales,
    colors,
    opacities,
    cam,
    background=None,
    mode: str = "exact",
    buckets=(16, 32, 64, 128, 256),
    budget_mb: float = 512,
    light_eps: float = 1e-3,
    keep_ratio: float = 0.75,
    max_pairs: int = 2_000_000,
):
    """Same inputs and outputs as gaussians3d.render(): image, depth, alpha, info.

    mode
      "exact"    every blob that reaches a tile is drawn, in as many passes as needed
      "sampled"  at most max(buckets) blobs per tile, one pass: keep_ratio of them chosen by
                 importance, the rest at random (approximate)
      "auto"     "exact" if there are at most max_pairs (blob, tile) pairs, else "sampled"
    buckets     padding sizes; a tile drawing 20 blobs is padded to 32, not to the largest.
                The largest is also the most blobs a tile draws in one pass.
    budget_mb   memory for blending one group of tiles; bigger = fewer, larger groups
    light_eps   a tile skips later passes once every pixel has less light left than this
    info        ids, mean2d (from project_blobs), visible (blobs drawn), mode used,
                pairs, passes
    """
    h, w = cam.height, cam.width
    nx, ny = math.ceil(w / TILE), math.ceil(h / TILE)
    if background is None:
        background = torch.zeros(3, dtype=means.dtype, device=means.device)

    ids, mean2d, z, cov, inv = project_blobs(means, quats, scales, cam)
    colors, opacities = colors[ids], opacities[ids]

    blob, tile = make_pairs(mean2d, z, cov, opacities, nx, ny)
    used = mode
    if mode == "auto":
        used = "exact" if len(blob) <= max_pairs else "sampled"
    if used == "sampled":
        blob, tile = select_pairs(blob, tile, cov, opacities, nx * ny, max(buckets), keep_ratio)
    elif used != "exact":
        raise ValueError(f"unknown mode {mode!r}")

    color, depth, light, drawn, passes = blend_pairs(
        blob, tile, mean2d, inv, colors, opacities, z, nx, ny, buckets, budget_mb, light_eps
    )

    def to_image(x):  # (tiles, PIX, ...) -> (H, W, ...)
        x = x.reshape(ny, nx, TILE, TILE, *x.shape[2:]).transpose(1, 2)
        return x.reshape(ny * TILE, nx * TILE, *x.shape[4:])[:h, :w]

    image = to_image(color + light[..., None] * background)
    info = {
        "ids": ids,
        "mean2d": mean2d,
        "visible": drawn,
        "mode": used,
        "pairs": len(blob),
        "passes": passes,
    }
    return image, to_image(depth), to_image(1 - light), info


def blob_reach(opacities):
    """k^2, where k = how many spreads from its centre a blob fades below 1/255.

    opacity * exp(-k^2 / 2) = 1/255  ->  k^2 = 2 ln(255 * opacity). Negative for blobs that
    are invisible everywhere. Strong blobs reach about 3.3 spreads, faint ones less.
    """
    return 2 * torch.log(opacities / MIN_ALPHA)


@torch.no_grad()
def make_pairs(mean2d, z, cov, opacities, nx, ny):
    """One (blob, tile) pair for every tile a blob reaches, sorted by tile, then near -> far."""
    a, b, c = cov
    n = len(z)
    widest = 0.5 * (a + c) + torch.sqrt(0.25 * (a - c) ** 2 + b * b)  # largest variance
    k2 = blob_reach(opacities)
    r = torch.sqrt(k2.clamp(min=0) * widest)  # radius in pixels

    # Range of tiles covered by the square (centre +- r).
    u, v = mean2d[:, 0], mean2d[:, 1]
    x0, x1 = torch.floor((u - r) / TILE).long(), torch.floor((u + r) / TILE).long()
    y0, y1 = torch.floor((v - r) / TILE).long(), torch.floor((v + r) / TILE).long()
    reach = (k2 > 0) & (x1 >= 0) & (x0 < nx) & (y1 >= 0) & (y0 < ny)
    x0, x1 = x0.clamp(0, nx - 1), x1.clamp(0, nx - 1)
    y0, y1 = y0.clamp(0, ny - 1), y1.clamp(0, ny - 1)
    width = x1 - x0 + 1
    n_tiles = torch.where(reach, width * (y1 - y0 + 1), torch.zeros_like(width))

    # Blob i gets n_tiles[i] pairs; number them 0..n_tiles[i]-1 and walk its tile rectangle.
    blob = torch.repeat_interleave(torch.arange(n, device=z.device), n_tiles)
    first = torch.cumsum(n_tiles, 0) - n_tiles  # index of each blob's first pair
    local = torch.arange(len(blob), device=z.device) - first[blob]
    tx = x0[blob] + local % width[blob]
    ty = y0[blob] + local // width[blob]
    tile = ty * nx + tx

    # Sort by tile, and inside a tile by depth: key = tile * n + depth rank.
    depth_rank = torch.empty(n, dtype=torch.long, device=z.device)
    depth_rank[torch.argsort(z)] = torch.arange(n, device=z.device)
    order = torch.argsort(tile * n + depth_rank[blob])
    return blob[order], tile[order]


@torch.no_grad()
def select_pairs(blob, tile, cov, opacities, n_tiles, slots, keep_ratio):
    """Mode "sampled": in tiles with more than `slots` blobs, keep only `slots` of them.

    keep_ratio of the slots go to the most important blobs, the rest to random others, so
    every blob keeps a chance to be drawn and get a gradient. Pairs stay in (tile, depth) order.

    Importance = opacity x share of the tile the blob covers x estimated light reaching it,
    where the light is dimmed by the blobs in front of it in the same tile.
    """
    counts = torch.bincount(tile, minlength=n_tiles)
    first = torch.cumsum(counts, 0) - counts  # index of each tile's first pair

    a, b, c = cov
    area = math.pi * blob_reach(opacities).clamp(min=0) * torch.sqrt(a * c - b * b)
    block = (opacities * (area / PIX).clamp(max=1))[blob]  # how much of the tile each blocks
    # Blocking by the blobs in front, inside the same tile: running sum minus the sum
    # before the tile started. (A float32 running sum drifts slightly; fine for a ranking.)
    run = torch.cumsum(block, 0)
    start = first[tile]
    in_front = run - block - (run[start] - block[start])
    importance = block * torch.exp(-in_front)

    crowded = counts[tile] > slots
    n_top = round(slots * keep_ratio)
    top = rank_in_tile(tile, importance, first) < n_top
    second = top.float() + torch.rand_like(importance)  # top ones first, then random order
    keep = ~crowded | (rank_in_tile(tile, second, first) < slots)
    return blob[keep], tile[keep]


def rank_in_tile(tile, score, first):
    """Position of each pair inside its tile when the tile is sorted by score, largest first."""
    n = len(score)
    every = torch.arange(n, device=score.device)
    score_rank = torch.empty_like(every)
    score_rank[torch.argsort(score, descending=True)] = every
    order = torch.argsort(tile * n + score_rank)
    rank = torch.empty_like(every)
    rank[order] = every - first[tile[order]]
    return rank


def blend_pairs(
    blob, tile, mean2d, inv, colors, opacities, z, nx, ny, buckets, budget_mb, light_eps
):
    """Front-to-back blending of the sorted pairs, in passes, buckets and groups.

    Returns color (T, PIX, 3), depth (T, PIX), light left (T, PIX), drawn (n,) bool, passes.
    """
    dev, dtype = z.device, z.dtype
    n_tiles = nx * ny
    sizes = torch.tensor(sorted(buckets), device=dev)
    chunk = max(buckets)  # most blobs a tile draws in one pass
    slots_per_group = max(1, int(budget_mb * 2**20 / (4 * TABLES * PIX)))  # (tile, blob) slots

    counts = torch.bincount(tile, minlength=n_tiles)
    pos = torch.arange(len(blob), device=dev) - (torch.cumsum(counts, 0) - counts)[tile]
    oy, ox = torch.meshgrid(torch.arange(TILE) + 0.5, torch.arange(TILE) + 0.5, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev, dtype), oy.reshape(-1).to(dev, dtype)  # (PIX,) inside a tile

    color = torch.zeros(n_tiles, PIX, 3, device=dev, dtype=dtype)
    depth = torch.zeros(n_tiles, PIX, device=dev, dtype=dtype)
    light = torch.ones(n_tiles, PIX, device=dev, dtype=dtype)
    drawn = torch.zeros(len(z), dtype=torch.bool, device=dev)
    A, B, C = inv

    passes = 0
    max_count = int(counts.max()) if len(blob) else 0
    for p in range(math.ceil(max_count / chunk)):
        with torch.no_grad():
            n_in = (counts - p * chunk).clamp(0, chunk)  # blobs each tile draws in this pass
            active = n_in > 0
            if p > 0:
                active &= light.amax(dim=1) > light_eps  # early stop: tile is nearly opaque
            if not active.any():
                break
            in_pass = (pos // chunk == p) & active[tile]
            b, t, slot = blob[in_pass], tile[in_pass], pos[in_pass] % chunk
            drawn[b] = True
            bucket = sizes[torch.searchsorted(sizes, n_in)]  # smallest size that fits
        passes += 1

        for K in sizes.tolist():
            with torch.no_grad():
                tiles_k = torch.nonzero(active & (bucket == K)).squeeze(1)
                if len(tiles_k) == 0:
                    continue
                row = torch.full((n_tiles,), -1, device=dev)
                row[tiles_k] = torch.arange(len(tiles_k), device=dev)
                mine = row[t] >= 0  # pairs whose tile is in this bucket
                table = torch.full((len(tiles_k), K), -1, device=dev)  # -1 = padding
                table[row[t[mine]], slot[mine]] = b[mine]

            group = max(1, slots_per_group // K)
            for g0 in range(0, len(tiles_k), group):
                tg, idx = tiles_k[g0 : g0 + group], table[g0 : g0 + group]
                real = (idx >= 0).to(dtype)
                idx = idx.clamp(min=0)
                px = ((tg % nx) * TILE).to(dtype)[:, None] + ox  # (g, PIX) pixel centres
                py = ((tg // nx) * TILE).to(dtype)[:, None] + oy
                args = (
                    px, py, mean2d[idx], A[idx], B[idx], C[idx],
                    opacities[idx] * real, colors[idx], z[idx], light[tg],
                )  # fmt: skip
                if torch.is_grad_enabled():
                    c_g, d_g, left = checkpoint(blend_group, *args, use_reentrant=False)
                else:
                    c_g, d_g, left = blend_group(*args)
                color = color.index_add(0, tg, c_g)
                depth = depth.index_add(0, tg, d_g)
                light = light.index_copy(0, tg, left)
    return color, depth, light, drawn, passes


def blend_group(px, py, mean2d, A, B, C, opacity, color, z, light_in):
    """render()'s steps 4-5 for g tiles with K blob slots each (padding has opacity 0).

    px, py (g, PIX); mean2d (g, K, 2); A, B, C, opacity, z (g, K); color (g, K, 3);
    light_in (g, PIX) = light left after earlier passes.
    Returns color (g, PIX, 3), depth (g, PIX), light left (g, PIX).
    """
    dx = px[:, None, :] - mean2d[..., 0:1]  # (g, K, PIX)
    dy = py[:, None, :] - mean2d[..., 1:2]
    power = -0.5 * (A[..., None] * dx**2 + 2 * B[..., None] * dx * dy + C[..., None] * dy**2)
    alpha = (opacity[..., None] * torch.exp(power)).clamp(max=MAX_ALPHA)

    left = torch.cumprod(1 - alpha, dim=1)  # light left after slots 0..k
    before = torch.cat([torch.ones_like(left[:, :1]), left[:, :-1]], dim=1)
    weight = alpha * before * light_in[:, None, :]
    color_out = torch.einsum("gkp,gkc->gpc", weight, color)
    depth_out = torch.einsum("gkp,gk->gp", weight, z)
    return color_out, depth_out, light_in * left[:, -1]
