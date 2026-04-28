from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark import Benchmark


@pytest.fixture
def mock_benchmark(mocker):
    mock_evaluator_cls = mocker.patch("benchmark.Evaluator")
    mock_evaluator_instance = MagicMock()
    mock_evaluator_instance.model_name = "MockModel"
    mock_evaluator_instance.upscale.return_value = "dummy_tensor"
    mock_evaluator_cls.return_value = mock_evaluator_instance

    mocker.patch("benchmark.Visualizer")

    bm = Benchmark(device="cpu")
    bm._init_evaluator = MagicMock(return_value=mock_evaluator_instance)  # type: ignore
    return bm


def test_benchmark_upscale(mock_benchmark: Benchmark, tmp_path: Path):
    config_paths = [Path("models/ModelA/config.yaml"), Path("models/ModelB/config.yaml")]
    input_img_paths = [Path("input/img1.png"), Path("input/img2.png")]
    output_img_paths = [tmp_path / "out1.png", tmp_path / "out2.png"]

    mock_benchmark.upscale(config_paths, input_img_paths, output_img_paths, downscale=False)

    assert mock_benchmark._init_evaluator.call_count == 2  # type: ignore

    eval_instance = mock_benchmark._init_evaluator.return_value  # type: ignore

    assert eval_instance.upscale.call_count == 4

    assert mock_benchmark.visualizer.save_image.call_count == 4  # type: ignore
