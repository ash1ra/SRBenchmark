from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from dataset import BenchmarkDataset


@pytest.fixture
def dummy_dataset(tmp_path: Path):
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x4"
    hr_dir.mkdir()
    lr_dir.mkdir()

    def make_img(path: Path, size: tuple):
        img = torch.randint(0, 255, size, dtype=torch.uint8)
        write_png(img, str(path))

    make_img(hr_dir / "img1.png", (3, 128, 128))
    make_img(lr_dir / "img1.jpg", (3, 32, 32))

    make_img(hr_dir / "img2.png", (3, 128, 128))

    make_img(lr_dir / "img3.png", (3, 32, 32))

    return tmp_path


def test_dataset_loading(dummy_dataset: Path):
    dataset = BenchmarkDataset(dummy_dataset, scaling_factor=4)
    assert len(dataset) == 1
    assert dataset.common_stems == ["img1"]


def test_dataset_getitem(dummy_dataset: Path):
    dataset = BenchmarkDataset(dummy_dataset, scaling_factor=4)
    item = dataset[0]

    assert "hr" in item and "lr" in item and "name" in item
    assert item["name"] == "img1"

    assert item["hr"].dtype == torch.float32  # type: ignore
    assert item["hr"].max() <= 1.0  # type: ignore
    assert item["hr"].min() >= 0.0  # type: ignore
