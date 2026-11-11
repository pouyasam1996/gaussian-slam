import pytest
import torch

from gaussian_slam import device as device_mod
from gaussian_slam.device import ENV_VAR, get_device


def _fake_hardware(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod, "_mps_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_preference_order(monkeypatch, cuda, mps, expected):
    monkeypatch.delenv(ENV_VAR, raising=False)
    _fake_hardware(monkeypatch, cuda=cuda, mps=mps)
    assert get_device().type == expected


def test_env_var_overrides_auto(monkeypatch):
    _fake_hardware(monkeypatch, cuda=True, mps=True)
    monkeypatch.setenv(ENV_VAR, "cpu")
    assert get_device() == torch.device("cpu")


def test_prefer_argument_beats_env_var(monkeypatch):
    _fake_hardware(monkeypatch, cuda=False, mps=True)
    monkeypatch.setenv(ENV_VAR, "cpu")
    assert get_device("mps").type == "mps"


def test_env_var_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "  CPU ")
    assert get_device() == torch.device("cpu")


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_unavailable_device_raises(monkeypatch, name):
    _fake_hardware(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="not available"):
        get_device(name)


def test_real_device_runs_a_tensor_op(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    dev = get_device()
    x = torch.ones(2, 3, device=dev)
    assert x.sum().item() == 6.0
