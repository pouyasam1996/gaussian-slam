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
