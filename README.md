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

The tutorial has three steps. Each has its own package and script folder, so earlier steps stay runnable as they were. Code used by every step lives at the top of the package.

```
src/gaussian_slam/
  device.py, geometry.py, metrics.py    shared by all steps
  step1_2d/              gaussians2d.py                 2D blobs fitted to a photo
  step2_3d/              gaussians3d.py                 3D blobs, simple dense renderer, simulated scenes
  step3_optimizations/   tiles.py                       exact tiled renderer
                         gaussians3d.py                 model with view-dependent colour, 3DGS training loop
                         densify.py                     clone, split, prune
                         blender.py                     Lego dataset loader
scripts/
  step1_2d/fit_2d.py
  step2_3d/synthetic_3d.py
  step3_optimizations/optimized_3dgaussian_run.py
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

## Step 3: optimizations, tested on the Lego scene

Step 2's renderer is correct but only works for small, simulated scenes. Step 3 takes the first real multi-view data, the Lego bulldozer from the NeRF-Synthetic (Blender) dataset (100 training and 200 test views at 800x800), and adds what is needed to fit it well:

1. **Tiled renderer** (`tiles.py`): the same picture as step 2's `render()`, with far less memory.
2. **The 3DGS training recipe** (`gaussians3d.py`): view-dependent colour, an L1 + SSIM loss, and a learning rate per parameter.
3. **Densification** (`densify.py`): add blobs where detail is missing and remove useless ones.

### Why a tiled renderer

Step 2 evaluates every blob at every pixel. Each intermediate value (pixel offset, power, alpha, light left, weight, ...) is a table of N blobs x H\*W pixels, and autograd keeps about a dozen of them for the backward pass:

| Scene | One table | All tables |
| --- | --- | --- |
| Lego, 100x100 px, 20 000 blobs | 0.8 GB | ~10 GB |
| Lego, 800x800 px, 20 000 blobs | 51 GB | ~600 GB |

Almost all of that is wasted. A blob is a few pixels wide, and about 3.3 spreads from its centre its alpha falls below 1/255, which is invisible on a screen. A typical Lego blob reaches 1 to 4 of the 49 tiles of a 100 px image. The tiled renderer computes only what is visible, in six stages:

| Stage | What it does | What it saves |
| --- | --- | --- |
| 1. project | Blobs to 2D ellipses, exactly as in step 2 | n/a |
| 2. pairs | Cut the image into 16x16 tiles; from the radius where a blob fades below 1/255, list the tiles it reaches as (blob, tile) pairs, sorted by tile, then near to far | Work is pairs x 256 pixels instead of all blobs x all pixels |
| 3. passes | Each pass draws the next 256 blobs of every tile, nearest first, carrying the light left from earlier passes; a tile with almost no light left stops | One table never holds more than 256 blobs per tile |
| 4. buckets | Tiles are grouped by how many blobs they draw and padded only to 16, 32, 64, 128 or 256 | Little work spent on padding |
| 5. groups | Each bucket is split into groups of tiles that fit `budget_mb` | Peak memory is fixed, whatever the blob count |
| 6. recompute | Each group's blending runs under `torch.utils.checkpoint`: forward keeps only the inputs, backward recomputes the tables | Memory no longer grows with the number of groups, for about 2x the compute of a group |

The result is exact: the blending math is step 2's, run on smaller tables. The only differences are the 1/255 cut-off and the early stop, which skips at most 0.001 of light. `optimized_3dgaussian_run.py` checks this before training: on a small scene it compares the image and the position gradients of both renderers, and stops with an `AssertionError` if they differ.

### The training recipe

These are the settings of the original 3D Gaussian Splatting method. The scale of our parameters matches theirs (opacity through a sigmoid, log-scales, quaternions, positions in world units), so their values carry over.

- **Learning rates per parameter**: positions 0.001, colour 0.0025, opacity 0.025, log-scales 0.005, rotations 0.001, view-dependent colour 0.0025 / 20. All decay along a cosine to 1% of their start value.
- **View-dependent colour**: each blob's colour is a base colour plus spherical harmonics up to degree 3, evaluated in the direction from the camera to the blob. Shiny or shaded surfaces then look different from different sides. One degree is switched on every 1000 steps, so the base colour settles first.
- **Loss**: 0.8 x L1 + 0.2 x (1 - SSIM). SSIM compares local mean, contrast and structure in an 11x11 window, which keeps edges sharp where a per-pixel error alone tends to blur.

### Densification

The model starts from random blobs in a cube (the dataset has no point cloud). Every 100 steps, blobs that are pulled hard on screen are cloned if small or split in two if large. A clone and its original each get opacity 1 - sqrt(1 - o), so together they block as much light as the original did. Nearly transparent or huge blobs are pruned. Every `--reset-every` steps all opacities drop to 0.01, so blobs that are not needed fade and get pruned. Adam's running averages follow each blob. Densification runs for the first 60% of training.

### Running it

`blender.py` reads `transforms_<split>.json`, converts each Blender/OpenGL camera to OpenCV, places the RGBA images on a **white** background and shrinks them by averaging pixel blocks. Training passes that white background to `fit()` and every render.

Put the dataset in `data/lego/` (ignored by git), then:

```bash
uv run python scripts/step3_optimizations/optimized_3dgaussian_run.py                 # 100 px, 10k steps
uv run python scripts/step3_optimizations/optimized_3dgaussian_run.py --steps 3000    # quicker check
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/lego` | Scene folder |
| `--size` | `100` | Image side in pixels (the data is 800) |
| `--n` | `10000` | Blobs at the start, random in a cube |
| `--steps` | `10000` | Training steps |
| `--extent` | `1.3` | Half-width of the start cube |
| `--lr-means`, `--lr-colors`, `--lr-opacity`, `--lr-scales`, `--lr-quats` | `0.001`, `0.0025`, `0.025`, `0.005`, `0.001` | Learning rates |
| `--sh-degree` | `3` | View-dependent colour; `0` = one colour per blob |
| `--ssim-weight` | `0.2` | Loss (1 - w) L1 + w (1 - SSIM) |
| `--no-densify` | off | Keep the blob count fixed |
| `--max-blobs` | `30000` | Blob cap for densification |
| `--grad-threshold` | `7.5e-6` | Densification threshold; set it near the `p90` printed at each densification (it depends on `--size` and the loss) |
| `--small-px` | `1.5` | Blobs smaller than this many pixels (at the object's distance) are cloned, larger ones split |
| `--reset-every` | `3000` | Opacity reset period; `0` = never |
| `--budget-mb` | `512` | Memory for blending one group of tiles (speed only, not the image) |
| `--eval-views` | `5` | Training views used for the PSNR printed every 100 steps |

Every 100 steps it prints the mean PSNR, the blob count, and the last step's (blob, tile) pairs and passes. The final score uses every training view and all 200 test views (PSNR and SSIM). Results go to `outputs/lego/`:

- `test_views.png`: 6 test views, target / fitted / error
- `training.png`: PSNR on `--eval-views` training views during training, with the final test PSNR
- `orbit.gif`: a full turn around the fitted model at 2x the training size
- `model.pt`: the fitted blobs (`model.state_dict()`)

| File | Purpose |
| --- | --- |
| `src/gaussian_slam/step3_optimizations/tiles.py` | `render_tiled()`: the exact tiled renderer, with the optimizations above |
| `src/gaussian_slam/step3_optimizations/gaussians3d.py` | `Gaussians3D` with view-dependent colour, and `fit()` with the 3DGS recipe and optional densification |
| `src/gaussian_slam/step3_optimizations/densify.py` | `DensityControl`: clone, split, prune and opacity reset |
| `src/gaussian_slam/step3_optimizations/blender.py` | `load_split()`: cameras and white-background images for one split |
| `src/gaussian_slam/metrics.py` | PSNR and SSIM |
| `scripts/step3_optimizations/optimized_3dgaussian_run.py` | Checks the tiled renderer against step 2, trains on Lego, scores the test views, saves the figures, orbit and model |

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
