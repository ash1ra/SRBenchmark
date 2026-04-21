from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class BenchmarkDataset(Dataset):
    def __init__(self, dataset_path: Path, scaling_factor: int) -> None:
        super().__init__()

        self.hr_dir_path = dataset_path / "HR"
        self.lr_dir_path = dataset_path / f"LR_x{scaling_factor}"

        if not self.hr_dir_path.exists() or not self.lr_dir_path.exists():
            raise FileNotFoundError(f"[Data] Datasets directories not found in '{dataset_path}'")

        valid_exts = {".png", ".jpg", ".jpeg"}
        hr_img_names = {p.name for p in self.hr_dir_path.iterdir() if p.suffix.lower() in valid_exts}
        lr_img_names = {p.name for p in self.lr_dir_path.iterdir() if p.suffix.lower() in valid_exts}

        self.img_names = sorted(list(hr_img_names & lr_img_names))

        if not self.img_names:
            raise FileNotFoundError(
                f"[Data] No matching files found between '{self.hr_dir_path}' and '{self.lr_dir_path}'"
            )

        if len(hr_img_names) != len(lr_img_names):
            print(
                f"[Data] Count mismatch! HR: {len(hr_img_names)}, LR: {len(lr_img_names)}. "
                f"Proceeding with {len(self.img_names)} common files."
            )

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        img_name = self.img_names[index]

        hr_img_path = self.hr_dir_path / img_name
        lr_img_path = self.lr_dir_path / img_name

        hr_img_tensor = read_image(str(hr_img_path), mode=ImageReadMode.RGB)
        lr_img_tensor = read_image(str(lr_img_path), mode=ImageReadMode.RGB)

        hr_img_tensor = hr_img_tensor.float() / 255.0
        lr_img_tensor = lr_img_tensor.float() / 255.0

        return {
            "hr": hr_img_tensor,
            "lr": lr_img_tensor,
            "name": Path(img_name).stem,
        }
