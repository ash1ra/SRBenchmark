import importlib
import warnings
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyiqa
import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image, write_png
from tqdm import tqdm

from dataset import BenchmarkDataset
from utils import calculate_psnr, calculate_ssim


class Evaluator:
    def __init__(
        self,
        config_path: Path,
        device: Literal["cpu", "cuda"] = "cpu",
        tile_size: int | None = None,
        tile_overlap: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 4,
    ) -> None:
        with open(config_path, "r", encoding="UTF-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)
        self.scaling_factor = self.config["model_params"]["scaling_factor"]
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._init_model(config_path)

    def _init_model(self, config_path: Path) -> None:
        self.model_name = config_path.parent.name
        self.module_path = f"models.{self.model_name}.model"

        self.model_class = getattr(importlib.import_module(self.module_path), self.model_name)
        self.model = cast(nn.Module, self.model_class(**self.config["model_params"])).to(self.device)

        self.weights_path = config_path.parent / self.config["weights_path"]

        if self.weights_path.suffix == ".pth":
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
        elif self.weights_path.suffix == ".safetensors":
            self.model.load_state_dict(load_file(self.weights_path, device=str(self.device)))

        self.model.eval()

    def _run_model_tiled(self, lr_img_tensor: Tensor) -> Tensor:
        assert self.tile_size is not None

        num_channels, lr_img_height, lr_img_width = lr_img_tensor.shape

        sr_img_height = lr_img_height * self.scaling_factor
        sr_img_width = lr_img_width * self.scaling_factor
        sr_img_shape = (num_channels, sr_img_height, sr_img_width)

        sr_accumulated_values = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")
        sr_weight_map = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")

        stride = self.tile_size - self.tile_overlap
        height_steps = list(range(0, lr_img_height - self.tile_size, stride)) + [lr_img_height - self.tile_size]
        width_steps = list(range(0, lr_img_width - self.tile_size, stride)) + [lr_img_width - self.tile_size]

        if lr_img_height < self.tile_size:
            height_steps = [0]

        if lr_img_width < self.tile_size:
            width_steps = [0]

        pbar = tqdm(total=len(height_steps) * len(width_steps), desc="Processing tiles", leave=False)

        for height_step in height_steps:
            for width_step in width_steps:
                lr_height_end = min(height_step + self.tile_size, lr_img_height)
                lr_width_end = min(width_step + self.tile_size, lr_img_width)
                lr_height_start = max(0, lr_height_end - self.tile_size)
                lr_width_start = max(0, lr_width_end - self.tile_size)

                lr_img_patch = lr_img_tensor[:, lr_height_start:lr_height_end, lr_width_start:lr_width_end]
                lr_img_patch = lr_img_patch.to(self.device)

                pad = 16
                lr_img_patch_padded = F.pad(lr_img_patch, (pad, pad, pad, pad), mode="reflect")

                sr_img_patch_padded = self.model(lr_img_patch_padded.unsqueeze(0)).squeeze(0).cpu()

                sr_pad = pad * self.scaling_factor
                sr_img_patch = sr_img_patch_padded[:, sr_pad:-sr_pad, sr_pad:-sr_pad]

                sr_height_end = lr_height_end * self.scaling_factor
                sr_width_end = lr_width_end * self.scaling_factor
                sr_height_start = lr_height_start * self.scaling_factor
                sr_width_start = lr_width_start * self.scaling_factor

                sr_accumulated_values[:, sr_height_start:sr_height_end, sr_width_start:sr_width_end] += sr_img_patch
                sr_weight_map[:, sr_height_start:sr_height_end, sr_width_start:sr_width_end] += 1.0

                pbar.update(1)
        pbar.close()

        return sr_accumulated_values.div_(sr_weight_map.clamp(min=1e-6)).to(self.device)

    def _run_model(self, lr_img_tensor: Tensor) -> Tensor:
        if self.tile_size is not None and self.tile_size > 0:
            return self._run_model_tiled(lr_img_tensor)

        window_size = self.config.get("model_params", {}).get("window_size", 16)

        _, lr_img_height, lr_img_width = lr_img_tensor.shape
        padding_right = (window_size - (lr_img_width % window_size)) % window_size
        padding_bottom = (window_size - (lr_img_height % window_size)) % window_size

        if padding_right > 0 or padding_bottom > 0:
            lr_tensor_padded = F.pad(lr_img_tensor, (0, padding_right, 0, padding_bottom), mode="reflect")
        else:
            lr_tensor_padded = lr_img_tensor

        sr_tensor_padded = self.model(lr_tensor_padded.unsqueeze(0)).squeeze(0)

        if padding_right > 0 or padding_bottom > 0:
            sr_img_tensor = sr_tensor_padded[
                :, : lr_img_height * self.scaling_factor, : lr_img_width * self.scaling_factor
            ]
        else:
            sr_img_tensor = sr_tensor_padded

        return sr_img_tensor

    def _preprocess_img(self, img_path: Path) -> Tensor:
        img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
        img_tensor = img_tensor.float() / 255.0

        return img_tensor.to(self.device)

    def _save_img(self, img_tensor: Tensor, save_path: Path) -> None:
        img_tensor = img_tensor.cpu().clamp(0, 1)
        img_tensor = (img_tensor * 255.0).to(torch.uint8)

        save_path.parent.mkdir(exist_ok=True, parents=True)
        write_png(img_tensor, str(save_path))

    def _init_dataloader(self, dataset_path: Path) -> DataLoader:
        dataset = BenchmarkDataset(dataset_path, scaling_factor=self.scaling_factor)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return dataloader

    def _evaluate_one_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str,
    ) -> dict[str, float]:
        scores = {"psnr": [], "ssim": [], "lpips": [], "clipiqa": [], "musiq": []}

        for batch in tqdm(
            dataloader, desc=f"Evaluating {self.model_name} on {dataset_name}", total=len(dataloader), leave=False
        ):
            lr_img_tensor = batch["lr"][0].to(device=self.device, non_blocking=True)
            hr_img_tensor = batch["hr"][0].to(device=self.device, non_blocking=True)

            sr_img_tensor = self._run_model(lr_img_tensor)
            sr_img_tensor = sr_img_tensor.float().clamp(0, 1)

            scores["psnr"].append(calculate_psnr(sr_img_tensor, hr_img_tensor, crop_border=self.scaling_factor))
            scores["ssim"].append(calculate_ssim(sr_img_tensor, hr_img_tensor, crop_border=self.scaling_factor))
            sr_img_batch = sr_img_tensor.unsqueeze(0)
            hr_img_batch = hr_img_tensor.unsqueeze(0)

            scores["lpips"].append(self.perceptual_metrics["lpips"](sr_img_batch, hr_img_batch).item())
            scores["clipiqa"].append(self.perceptual_metrics["clipiqa"](sr_img_batch).item())
            scores["musiq"].append(self.perceptual_metrics["musiq"](sr_img_batch).item())

        return {
            "PSNR": float(np.mean(scores["psnr"])),
            "SSIM": float(np.mean(scores["ssim"])),
            "LPIPS": float(np.mean(scores["lpips"])),
            "CLIPIQA": float(np.mean(scores["clipiqa"])),
            "MUSIQ": float(np.mean(scores["musiq"])),
        }

    def _print_results_table(self, results: dict[str, dict[str, float]]) -> None:
        print(f"\n{'=' * 82}")
        print(f"{'Benchmark: ' + self.model_name:^82}")
        print(f"{'-' * 82}")
        print(
            f"{'Dataset':<15} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10} | {'LPIPS (↓)':<10} | {'CLIPIQA (↑)':<11} | {'MUSIQ (↑)':<10}"
        )
        print(f"{'-' * 82}")

        for dataset_name, metrics in results.items():
            print(
                f"{dataset_name:<15} | "
                f"{metrics['PSNR']:>10.2f} | "
                f"{metrics['SSIM']:>10.4f} | "
                f"{metrics['LPIPS']:>10.4f} | "
                f"{metrics['CLIPIQA']:>11.4f} | "
                f"{metrics['MUSIQ']:>10.2f}"
            )

        print(f"{'=' * 82}\n")

    @torch.inference_mode()
    def evaluate(self, dataset_paths: list[Path]) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)

        self.perceptual_metrics = {
            "lpips": pyiqa.create_metric("lpips", as_loss=False, device=self.device),
            "clipiqa": pyiqa.create_metric("clipiqa", as_loss=False, device=self.device),
            "musiq": pyiqa.create_metric("musiq", as_loss=False, device=self.device),
        }

        warnings.filterwarnings("default", category=UserWarning)

        results = {}

        for dataset_path in dataset_paths:
            dataloader = self._init_dataloader(dataset_path)
            results[dataset_path.name] = self._evaluate_one_dataset(dataloader, dataset_path.name)

        self._print_results_table(results)

    @torch.inference_mode()
    def upscale(self, img_path: Path, output_path: Path) -> None:
        lr_img_tensor = self._preprocess_img(img_path)
        sr_img_tensor = self._run_model(lr_img_tensor)
        self._save_img(sr_img_tensor, output_path)

    @torch.inference_mode()
    def upscale_downscaled(self, img_path: Path, output_path: Path) -> None:
        hr_img_tensor = self._preprocess_img(img_path)
        _, hr_img_height, hr_img_width = hr_img_tensor.shape

        lr_img_height = hr_img_height // self.scaling_factor
        lr_img_width = hr_img_width // self.scaling_factor

        lr_img_tensor = F.interpolate(
            hr_img_tensor.unsqueeze(0),
            size=(lr_img_height, lr_img_width),
            mode="bicubic",
            antialias=True,
        ).clamp(0, 1)

        print(lr_img_tensor.min(), lr_img_tensor.max())

        sr_img_tensor = self._run_model(lr_img_tensor.squeeze(0))

        self._save_img(sr_img_tensor, output_path)


if __name__ == "__main__":
    evaluator = Evaluator(
        config_path=Path("models/HAT/config.yaml"),
        device="cuda",
        tile_size=256,
    )

    evaluator.upscale_downscaled(Path("images/hr_img_1.jpg"), Path("results/sr_img_1.png"))
    # evaluator.evaluate(
    #     [Path("data/Set5"), Path("data/Set14"), Path("data/BSDS100"), Path("data/Urban100"), Path("data/Manga109")]
    # )
