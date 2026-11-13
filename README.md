# gaussian-slam

SLAM with 3D Gaussian Splatting: incrementally reconstruct a scene as a set of 3D Gaussians while tracking the camera pose, to produce scenes and data for training.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
uv run pytest -v
```

Tests marked `gpu` are skipped automatically when no CUDA GPU is available.

## Layout

Each step has its own package and script folder, so earlier steps stay runnable as they were. Code used by every step lives at the top of the package.

```
src/gaussian_slam/
  device.py, geometry.py, metrics.py    shared by all steps
  step1_2d/            gaussians2d.py
  step2_3d/            gaussians3d.py  (dense renderer)
  step3_3d_advanced/   gaussians3d.py  (dense or tiled renderer), tiles.py, densify.py
  step4_lego/          blender.py      (dataset loader; reuses step 3's model and renderer)
scripts/
  step1_2d/fit_2d.py
  step2_3d/synthetic_3d.py
  step3_3d_advanced/synthetic_3d.py
  step4_lego/lego.py
```

## Step 1: Fitting 2D Gaussians to a photo

Before moving to 3D, this step approximates a single image with a set of colored 2D Gaussian blobs, optimized with gradient descent.

📖 Tutorial: [2D Gaussian Fitting Tutorial](https://medium.com/@samandi.pouya/2d-gaussian-fitting-tutorial-f415fd9d93c3) (Medium)

```bash
uv run python scripts/step1_2d/fit_2d.py path/to/photo.jpg
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--size` | `256` | Longer side of the resized image, in pixels |
| `--n` | `2000` | Number of Gaussians |
| `--steps` | `2000` | Number of optimization steps |

Training uses Adam with a cosine learning-rate decay and prints PSNR every 50 steps. Results are written to `outputs/fit2d/`:

- `target.png`: the resized input image
- `final.png`: the image rendered from the fitted Gaussians
- `psnr.png`: PSNR over training steps

Press **Ctrl+C** to stop early. The current result is still saved, but because the learning rate hasn't finished decaying it will be less refined than a full run.

| File | Purpose |
| --- | --- |
| `src/gaussian_slam/step1_2d/gaussians2d.py` | `Gaussians2D` model and the `fit()` training loop |
| `src/gaussian_slam/metrics.py` | PSNR |
| `src/gaussian_slam/device.py` | Picks the device: CUDA, then Apple MPS, then CPU (override with `GS_DEVICE=cpu`) |
| `scripts/step1_2d/fit_2d.py` | Command-line script to fit a photo |

## Step 2: 3D Gaussians from several cameras

This step moves to 3D. A scene is a set of 3D Gaussian blobs, each with a position, rotation, size, colour and opacity. To render it from a camera, every blob is projected to a 2D ellipse on the image, the blobs are sorted near to far, and their colours are blended front to back. The renderer is differentiable, so the blobs can be fitted to images taken from several cameras.

`scripts/step2_3d/synthetic_3d.py` tests the whole pipeline on a scene where the answer is known. It builds a true scene of blobs, renders it from 12 training cameras (two rings, one above and one below), scrambles every parameter and optimizes the scene back from the training images alone. 2 held-out cameras, at an angle no training camera has, show whether it learned the 3D shape or only memorized the training views.

```bash
uv run python scripts/step2_3d/synthetic_3d.py                  # a rainbow torus
uv run python scripts/step2_3d/synthetic_3d.py --shape helix
uv run python scripts/step2_3d/synthetic_3d.py --shape random   # blobs scattered in a cube
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--shape` | `torus` | True scene: `torus`, `helix` or `random` |
| `--size` | `64` | Image side, in pixels |
| `--n` | `300` | Number of blobs |
| `--steps` | `3000` | Number of optimization steps |

Before fitting, the script runs three quick checks: a point projected by hand, camera conversions (`look_at` and a Blender camera), and the renderer's gradients against small-step estimates. Any failure stops the script.

Each step trains on one random camera. Training uses Adam with a cosine learning-rate decay, and blob positions get a 10x smaller learning rate than the other parameters. Mean PSNR is printed every 100 steps. At the end the script prints training and held-out PSNR, and the **shape error**: the mean distance from each visible blob to the nearest true blob (the scene is about 2 units wide). Results are written to `outputs/synthetic3d/`:

- `scene_3d.png`: blob centres in 3D for the true scene, the scrambled start and the fitted result
- `views.png`: target, start, fitted, error and depth images for 4 training and both held-out cameras
- `training.png`: mean training PSNR over steps
- `heldout.png`: the held-out cameras rendered at 3x resolution with a zoom lens, true vs fitted

Press **Ctrl+C** to stop early, as in step 1.

Images are small (64 px) because, as in step 1, every blob is evaluated at every pixel.

### Background colour

`render()`, `Gaussians3D.forward()` and `fit()` take an optional `background` colour of shape `(3,)`: what shows through where no blob covers the pixel. It defaults to **black**, which is what `synthetic_3d.py` uses. The Blender dataset in the next step has a **white** background, so code for it must pass the background explicitly, without changing the default:

```python
white = torch.ones(3, device=device)
fit(model, cams, targets, background=white)
image, depth, alpha = model(cam, white)
```

If the background does not match the targets, the blobs are pushed to paint the background themselves.

| File | Purpose |
| --- | --- |
| `src/gaussian_slam/geometry.py` | Quaternion rotations, 3D covariances, the pinhole `Camera` (OpenCV convention), projection, `look_at`, and Blender/OpenGL to OpenCV camera conversion |
| `src/gaussian_slam/step2_3d/gaussians3d.py` | The `render()` function (image, depth, alpha), the `Gaussians3D` model and the multi-view `fit()` loop |
| `scripts/step2_3d/synthetic_3d.py` | Checks, then fits a scrambled synthetic scene and saves the figures |

## Step 3: tiled renderer and densification

Step 2 evaluates every blob at every pixel, which runs out of memory beyond small images. Step 3 keeps the same model and adds two things:

- **Tiled renderer** (`tiles.py`). The image is cut into 16x16 tiles, and each blob is drawn only in the tiles it reaches before fading below 1/255. Tiles are blended in passes and groups sized to a memory budget, and recomputed during the backward pass instead of stored. Modes: `exact` draws every blob (same picture as dense, up to the 1/255 cut-off); `sampled` keeps at most 256 blobs per crowded tile, mostly the important ones plus a few random (approximate, faster); `auto` picks `exact` unless there are too many (blob, tile) pairs.
- **Densification** (`densify.py`). The model starts with few blobs. Blobs that are pulled hard on screen are cloned (small) or split in two (large). Nearly transparent or huge blobs are pruned. Every `reset_every` steps all opacities are reset to 0.01, so unneeded blobs fade and get pruned. Adam's running averages follow each blob.

```bash
uv run python scripts/step3_3d_advanced/synthetic_3d.py --mode exact
uv run python scripts/step3_3d_advanced/synthetic_3d.py --mode exact --densify --grad-threshold 1.5e-5
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `dense` | Renderer for training: `dense`, `exact`, `sampled` or `auto`. Evaluation uses `exact` unless `dense` |
| `--densify` | off | Start from `--start-n` blobs and grow them |
| `--start-n` | `50` | Blobs at the start with `--densify` |
| `--max-blobs` | `2000` | Blob cap with `--densify` |
| `--grad-threshold` | `2e-4` | Screen gradient above which a blob is cloned or split |

`--shape`, `--size`, `--n` and `--steps` are as in step 2. Before fitting, the script also checks that dense and tiled renderers give the same image and gradients. Results go to `outputs/synthetic3d_advanced/`.

The gradient threshold depends on the image size, because the loss is averaged over pixels. At 64 px the default `2e-4` is above every blob's gradient, so nothing densifies; use about `1.5e-5`. Each densification prints the gradient percentiles to help tune it.

| File | Purpose |
| --- | --- |
| `src/gaussian_slam/step3_3d_advanced/gaussians3d.py` | `project_blobs()`, dense `render()`, `Gaussians3D` with a `mode` switch, and `fit()` with optional densification |
| `src/gaussian_slam/step3_3d_advanced/tiles.py` | `render_tiled()`: the tiled renderer |
| `src/gaussian_slam/step3_3d_advanced/densify.py` | `DensityControl`: clone, split, prune and opacity reset |
| `scripts/step3_3d_advanced/synthetic_3d.py` | Step 2's experiment with a renderer choice and densification |

## Step 4: the Lego scene

The first real multi-view data: the Lego bulldozer from the NeRF-Synthetic (Blender) dataset, 100 training and 200 test views at 800x800. Step 4 adds only a loader and a script; the model, renderers and densification are step 3's.

`blender.py` reads `transforms_<split>.json`, converts each Blender/OpenGL camera to OpenCV, places the RGBA images on a **white** background and shrinks them by averaging pixel blocks. Training passes that white background to `fit()` and every render.

Put the dataset in `data/lego/` (ignored by git), then:

```bash
uv run python scripts/step4_lego/lego.py                  # 100 px, 10k steps, exact, densify
uv run python scripts/step4_lego/lego.py --steps 3000     # quicker check
uv run python scripts/step4_lego/lego.py --mode sampled   # approximate, faster training steps
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/lego` | Scene folder |
| `--size` | `100` | Image side in pixels (the data is 800) |
| `--n` | `5000` | Blobs at the start, random in a cube (the dataset has no point cloud) |
| `--steps` | `10000` | Training steps |
| `--mode` | `exact` | Renderer for training steps |
| `--exact-every` | `10` | With `sampled` / `auto`, every this many steps is exact |
| `--keep-ratio` | `0.3 0.9` | `sampled` / `auto` steps: share of a crowded tile's slots given to the most important blobs, growing from start to end over training; the rest are random |
| `--no-densify` | off | Keep the blob count fixed |
| `--max-blobs` | `30000` | Blob cap for densification |
| `--grad-threshold` | `2.5e-6` | Densification threshold; set it near the `p90` printed at each densification (it depends on `--size`) |
| `--small-px` | `1.5` | Blobs smaller than this many pixels (at the object's distance) are cloned, larger ones split |
| `--budget-mb` | `512` | Memory for blending one group of tiles (speed only, not the image) |
| `--extent` | `1.3` | Half-width of the start cube |
| `--eval-views` | `5` | Training views used for the PSNR printed every 100 steps |

The final score uses every training view and all 200 test views, rendered with the exact renderer. Results go to `outputs/lego/`:

- `test_views.png`: 6 test views, target / fitted / error
- `training.png`: PSNR on `--eval-views` training views during training, with the final test PSNR
- `orbit.gif`: a full turn around the fitted model at 2x the training size
- `model.pt`: the fitted blobs (`model.state_dict()`)

| File | Purpose |
| --- | --- |
| `src/gaussian_slam/step4_lego/blender.py` | `load_split()`: cameras and white-background images for one split |
| `scripts/step4_lego/lego.py` | Trains on Lego, scores the test views, saves the figures, orbit and model |

## Development

| File | Purpose |
| --- | --- |
| `.github/workflows/ci.yml` | Lint, format check, tests on every PR |
| `.pre-commit-config.yaml` | Local checks before each commit |
| `pyproject.toml` | Dependencies, ruff and pytest settings |
| `tests/conftest.py` | Skips `gpu`-marked tests without CUDA |

`main` is protected: changes go through pull requests and must pass CI (`test`).

## License

[MIT](LICENSE)
