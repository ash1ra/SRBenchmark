from pathlib import Path

import pytest
import torch
from torch import nn

from evaluator import Evaluator


class DummySRModel(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


@pytest.fixture
def mock_evaluator(tmp_path: Path, mocker):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        f.write("model_params:\n  scaling_factor: 2\n  window_size: 4\n")

    mocker.patch("evaluator.Evaluator._init_model", autospec=True)
    mocker.patch("evaluator.Evaluator._init_perceptual_metrics", autospec=True)

    evaluator = Evaluator(config_path=config_file, device="cpu")

    evaluator.model = DummySRModel()
    evaluator.model_name = "DummyModel"
    evaluator.perceptual_metrics = {
        "lpips": lambda _x, _y: torch.tensor(0.1),
        "clipiqa": lambda _x: torch.tensor(0.8),
        "musiq": lambda _x: torch.tensor(50.0),
    }

    return evaluator


def test_evaluator_run_model(mock_evaluator: Evaluator):
    lr_tensor = torch.rand(3, 16, 16)

    sr_tensor = mock_evaluator._run_model(lr_tensor)

    assert sr_tensor.shape == (3, 32, 32)
    assert sr_tensor.dtype == torch.float32


def test_evaluator_upscale_downscaled(mock_evaluator: Evaluator):
    hr_img = torch.rand(3, 32, 32)
    sr_img = mock_evaluator.upscale_downscaled(hr_img)

    assert sr_img.shape == (3, 32, 32)


def test_get_model_stats_override(mock_evaluator: Evaluator):
    mock_evaluator.model_name = "ResShift"
    stats = mock_evaluator.get_model_stats()

    assert stats["model_name"] == "ResShift"
    assert stats["params"] == 118.59 * 1e6

    assert isinstance(stats["flops"], (float, int))
    assert stats["flops"] >= 0


def test_get_model_stats_standard(mock_evaluator: Evaluator, mocker):
    mock_evaluator.model_name = "UnknownModel"

    mocker.patch(
        "evaluator.get_model_complexity_info",
        return_value=(500000000.0, 2000000.0),
    )

    stats = mock_evaluator.get_model_stats()

    assert stats["model_name"] == "UnknownModel"
    assert stats["params"] == 2000000.0
    assert stats["flops"] == 1000000000.0  # FLOPs = MACs * 2
