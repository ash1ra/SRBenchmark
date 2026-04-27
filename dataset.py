from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from logger import logger


class BenchmarkDataset(Dataset):
    def __init__(self, dataset_path: Path, scaling_factor: int) -> None:
        super().__init__()

        self.hr_dir_path = dataset_path / "HR"
        self.lr_dir_path = dataset_path / f"LR_x{scaling_factor}"

        if not self.hr_dir_path.exists() or not self.lr_dir_path.exists():
            raise FileNotFoundError(f"[Data] Datasets directories not found in '{dataset_path}'")

        valid_exts = {".png", ".jpg", ".jpeg"}
        hr_files = {p.stem: p for p in self.hr_dir_path.iterdir() if p.suffix.lower() in valid_exts}
        lr_files = {p.stem: p for p in self.lr_dir_path.iterdir() if p.suffix.lower() in valid_exts}

        self.common_stems = sorted(list(set(hr_files.keys()) & set(lr_files.keys())))

        if not self.common_stems:
            raise FileNotFoundError(
                f"[Data] No matching files found between '{self.hr_dir_path}' and '{self.lr_dir_path}'"
            )

        if len(hr_files) != len(lr_files):
            logger.warning(
                f"[Data] Count mismatch! HR: {len(hr_files)}, LR: {len(lr_files)}. "
                f"Proceeding with {len(self.common_stems)} common files."
            )

        self.hr_paths = [hr_files[stem] for stem in self.common_stems]
        self.lr_paths = [lr_files[stem] for stem in self.common_stems]

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        hr_img_path = self.hr_paths[index]
        lr_img_path = self.lr_paths[index]

        hr_img_tensor = (read_image(str(hr_img_path), mode=ImageReadMode.RGB)).float() / 255.0
        lr_img_tensor = (read_image(str(lr_img_path), mode=ImageReadMode.RGB)).float() / 255.0

        return {
            "hr": hr_img_tensor,
            "lr": lr_img_tensor,
            "name": self.common_stems[index],
        }
