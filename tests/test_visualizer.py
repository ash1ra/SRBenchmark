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


def test_save_benchmark_excel(tmp_path: Path, sample_results):
    visualizer = Visualizer()
    output_dir = tmp_path / "results"

    visualizer.save_benchmark_excel(sample_results, output_dir)

    excel_file = output_dir / "benchmark_results.xlsx"
    assert excel_file.exists()

    sheets = pd.read_excel(excel_file, sheet_name=None)

    assert list(sheets.keys()) == ["Set5", "Set14"]

    df_set5 = sheets["Set5"]

    expected_columns = ["Model", "Params (M)", "Time", "PSNR", "SSIM"]
    assert list(df_set5.columns) == expected_columns

    assert len(df_set5) == 2

    assert df_set5.iloc[0]["Model"] == "Model_A"
    assert df_set5.iloc[0]["PSNR"] == 32.0
    assert df_set5.iloc[1]["Model"] == "Model_B"
    assert df_set5.iloc[1]["PSNR"] == 34.0


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
