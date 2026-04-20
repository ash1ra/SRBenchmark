import importlib
from pathlib import Path
from typing import Literal

import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.io import ImageReadMode, read_image, write_png


class Evaluator:
    def __init__(self, config_dir_path: Path, device: Literal["cpu", "cuda"] = "cpu") -> None:
        self.device = torch.device(device)

        with open(config_dir_path, "r", encoding="UTF-8") as f:
            self.config = yaml.safe_load(f)

        self.model_name = config_dir_path.parent.name
        self.module_path = f"models.{self.model_name}.model"

        self.model_class = getattr(importlib.import_module(self.module_path), self.model_name)
        self.model: nn.Module = self.model_class(**self.config["model_params"]).to(self.device)

        self.weights_path = config_dir_path.parent / self.config["weights_path"]

        if self.weights_path.suffix == ".pth":
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
        elif self.weights_path.suffix == ".safetensors":
            self.model.load_state_dict(load_file(self.weights_path, device=str(self.device)))

        self.model.eval()

        self.scaling_factor = self.config["model_params"]["scaling_factor"]

    def _preprocess_img(self, img_path: Path) -> Tensor:
        img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
        img_tensor = (img_tensor.float() / 255.0).unsqueeze(0)

        return img_tensor.to(self.device)

    def _save_img(self, img_tensor: Tensor, save_path: Path) -> None:
        img_tensor = img_tensor.squeeze(0).cpu().clamp(0, 1)
        img_tensor = (img_tensor * 255.0).to(torch.uint8)

        save_path.parent.mkdir(exist_ok=True, parents=True)
        write_png(img_tensor, str(save_path))

    def upscale(self, img_path: str | Path, output_path: str | Path) -> None:
        lr_img_tensor = self._preprocess_img(Path(img_path))
        sr_img_tensor = self.model(lr_img_tensor)
        self._save_img(sr_img_tensor, Path(output_path))


if __name__ == "__main__":
    evaluator = Evaluator(
        config_dir_path=Path("models/SRResNet/config.yaml"),
        device="cuda",
    )

    evaluator.upscale("images/butterfly.png", "results/sr_butterfly.png")
