import pytest
import torch

import gaussian_slam


def test_package_imports() -> None:
    assert gaussian_slam.__name__ == "gaussian_slam"


def test_tensor_sum():
    x = torch.ones(2, 3)
    assert x.sum().item() == 6.0


@pytest.mark.gpu
def test_cuda_tensor_sum():
    x = torch.ones(2, 3, device="cuda")
    assert x.sum().item() == 6.0
