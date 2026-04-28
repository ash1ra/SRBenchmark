import importlib
import io
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyiqa
import torch
from ptflops import get_model_complexity_info
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

from core_utils import ModelConfig, calculate_psnr, calculate_ssim, imresize
from dataset import BenchmarkDataset


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
        self.config = ModelConfig.from_yaml(config_path)

        self.device = torch.device(device)
        self.scaling_factor = self.config.scaling_factor
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._init_model(config_path)
        self._init_perceptual_metrics()

        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.bfloat16

    def _init_model(self, config_path: Path) -> None:
        self.model_name = config_path.parent.name
        self.module_path = f"models.{self.model_name}.model"

        self.model_class = getattr(importlib.import_module(self.module_path), self.model_name)
        self.model = cast(nn.Module, self.model_class(**self.config.model_params)).to(self.device)

        self.weights_path = config_path.parent / self.config.weights_path

        if self.weights_path.suffix in [".pt", ".pth"]:
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
        elif self.weights_path.suffix == ".safetensors":
            self.model.load_state_dict(load_file(self.weights_path, device=str(self.device)))

        self.model.eval()

    def _init_perceptual_metrics(self) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)

        with redirect_stdout(io.StringIO()):
            self.perceptual_metrics = {
                "lpips": pyiqa.create_metric("lpips", as_loss=False, device=self.device),
                "clipiqa": pyiqa.create_metric("clipiqa", as_loss=False, device=self.device),
                "musiq": pyiqa.create_metric("musiq", as_loss=False, device=self.device),
            }

        warnings.filterwarnings("default", category=UserWarning)

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
                lr_img_patch_padded = F.pad(lr_img_patch, (pad, pad, pad, pad), mode="replicate")

                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    sr_img_patch_padded = self.model(lr_img_patch_padded.unsqueeze(0)).squeeze(0).cpu()

                sr_img_patch_padded = sr_img_patch_padded.float()

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

        window_size = self.config.window_size

        _, lr_img_height, lr_img_width = lr_img_tensor.shape
        padding_right = (window_size - (lr_img_width % window_size)) % window_size
        padding_bottom = (window_size - (lr_img_height % window_size)) % window_size

        if padding_right > 0 or padding_bottom > 0:
            lr_tensor_padded = F.pad(lr_img_tensor, (0, padding_right, 0, padding_bottom), mode="replicate")
        else:
            lr_tensor_padded = lr_img_tensor

        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            sr_tensor_padded = self.model(lr_tensor_padded.unsqueeze(0)).squeeze(0)

        sr_tensor_padded = sr_tensor_padded.float()

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
        scores = {"psnr": [], "ssim": [], "lpips": [], "clipiqa": [], "musiq": [], "time": []}

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        is_cuda = self.device.type == "cuda"

        for batch in tqdm(
            dataloader, desc=f"Evaluating {self.model_name} on {dataset_name}", total=len(dataloader), leave=False
        ):
            lr_img_tensor = batch["lr"][0].to(device=self.device, non_blocking=True)
            hr_img_tensor = batch["hr"][0].to(device=self.device, non_blocking=True)

            if is_cuda:
                torch.cuda.synchronize()
                starter.record()
            else:
                start_time = time.perf_counter()

            sr_img_tensor = self._run_model(lr_img_tensor)

            if is_cuda:
                ender.record()
                torch.cuda.synchronize()
                inference_time = starter.elapsed_time(ender) / 1000.0
            else:
                inference_time = time.perf_counter() - start_time

            scores["time"].append(inference_time)

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
            "Time": float(np.sum(scores["time"])),
        }

    @torch.inference_mode()
    def evaluate(self, dataset_paths: list[Path]) -> dict[str, dict[str, float]]:
        results = {}

        for dataset_path in dataset_paths:
            dataloader = self._init_dataloader(dataset_path)
            results[dataset_path.name] = self._evaluate_one_dataset(dataloader, dataset_path.name)

        return results

    @torch.inference_mode()
    def upscale(self, img_tensor: Tensor) -> Tensor:
        return self._run_model(img_tensor.to(self.device))

    @torch.inference_mode()
    def upscale_downscaled(self, img_tensor: Tensor) -> Tensor:
        lr_img_tensor = imresize(img_tensor, scaling_factor=1 / self.scaling_factor, antialiasing=True).to(self.device)

        return self._run_model(lr_img_tensor.squeeze(0))

    def get_model_stats(self, input_shape: tuple[int, int, int] = (3, 64, 64)) -> dict[str, str | float]:
        param_overrides = {
            "ResShift": 118.59 * 1e6,
        }

        try:
            macs, params = get_model_complexity_info(
                model=self.model,
                input_res=input_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )

            if self.model_name in param_overrides:
                params = param_overrides[self.model_name]

            return {"model_name": self.model_name, "params": params, "flops": macs * 2}  # type: ignore

        except Exception:
            fallback_params = param_overrides.get(self.model_name, 0.0)
            return {"model_name": self.model_name, "params": fallback_params, "flops": 0.0}
