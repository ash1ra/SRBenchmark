import pytest
import torch
import pandas as pd
from pathlib import Path

from visualizer import Visualizer


@pytest.fixture
def sample_results():
    return {
        "Model_A": {
            "Set5": {"PSNR": 32.0, "SSIM": 0.90, "Params (M)": 1.5, "Time": 0.1},
            "Set14": {"PSNR": 28.0, "SSIM": 0.80, "Params (M)": 1.5, "Time": 0.12},
        },
        "Model_B": {
            "Set5": {"PSNR": 34.0, "SSIM": 0.92, "Params (M)": 16.0, "Time": 0.5},
            "Set14": {"PSNR": 30.0, "SSIM": 0.85, "Params (M)": 16.0, "Time": 0.55},
        },
    }


def test_save_benchmark_csv(tmp_path: Path, sample_results):
    visualizer = Visualizer()
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    visualizer.save_benchmark_csv(sample_results, output_dir)

    csv_file = output_dir / "benchmark.csv"
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == ["Model", "Dataset", "Params (M)", "Time", "PSNR", "SSIM"]
    assert len(df) == 4


def test_generate_plots(tmp_path: Path, sample_results):
    visualizer = Visualizer()
    output_dir = tmp_path / "results"

    visualizer.generate_plots(sample_results, output_dir)

    plots_dir = output_dir / "plots"
    assert plots_dir.exists()

    assert (plots_dir / "scatter_tradeoff_PSNR.png").exists()
    assert (plots_dir / "radar_balance.png").exists()
    assert (plots_dir / "bar_stability_PSNR.png").exists()


def test_get_crop_center_mocked(mocker):
    visualizer = Visualizer()
    dummy_tensor = torch.rand(3, 100, 100)

    mocker.patch("cv2.namedWindow")
    mocker.patch("cv2.setMouseCallback")
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("cv2.waitKey", return_value=13)

    center = visualizer.get_crop_center(dummy_tensor, crop_size=32)

    assert center == (50, 50)
