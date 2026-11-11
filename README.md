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

## Step 1: Fitting 2D Gaussians to a photo

Before moving to 3D, this step approximates a single image with a set of colored 2D Gaussian blobs, optimized with gradient descent.

📖 Tutorial: [2D Gaussian Fitting Tutorial](https://medium.com/@samandi.pouya/2d-gaussian-fitting-tutorial-f415fd9d93c3) (Medium)

```bash
uv run python scripts/fit_2d.py path/to/photo.jpg
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
| `src/gaussian_slam/gaussians2d.py` | `Gaussians2D` model and the `fit()` training loop |
| `src/gaussian_slam/metrics.py` | PSNR |
| `src/gaussian_slam/device.py` | Picks the device: CUDA, then Apple MPS, then CPU (override with `GS_DEVICE=cpu`) |
| `scripts/fit_2d.py` | Command-line script to fit a photo |

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
