import gc
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch.nn import functional as F

from evaluator import Evaluator
from utils import imresize_tensor
from visualizer import Visualizer


class Benchmark:
    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        tile_size: int | None = None,
        tile_overlap: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 4,
    ) -> None:
        self.device = device

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.visualizer = Visualizer()

    def _init_evaluator(self, config_path: Path) -> Evaluator:
        return Evaluator(
            config_path=config_path,
            device=self.device,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def evaluate(
        self,
        config_paths: list[Path],
        dataset_paths: list[Path],
        save_csv_path: Path | None = None,
    ) -> None:
        results = {}

        for config_path in config_paths:
            evaluator = self._init_evaluator(config_path)
            model_name = evaluator.model_name

            print(f"\n[{model_name}] | Starting benchmark...")

            model_results = evaluator.evaluate(dataset_paths)
            results[model_name] = model_results

            self.visualizer.print_results_table(model_name, model_results)

            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

        if save_csv_path and results:
            self.visualizer.save_benchmark_csv(results, save_csv_path)

    def upscale(
        self,
        config_path: Path,
        input_img_path: Path,
        output_img_path: Path,
        downscale: bool = False,
    ) -> None:
        img_tensor = self.visualizer.read_image(input_img_path)

        evaluator = self._init_evaluator(config_path)

        if downscale:
            sr_img_tensor = evaluator.upscale_downscaled(img_tensor)
        else:
            sr_img_tensor = evaluator.upscale(img_tensor)

        self.visualizer.save_image(sr_img_tensor, output_img_path)

        del evaluator
        gc.collect()
        torch.cuda.empty_cache()

    def compare(
        self,
        config_paths: list[Path],
        hr_img_path: Path,
        output_img_path: Path,
        lr_img_path: Path | None = None,
        crop_size: int | None = None,
    ) -> None:
        hr_img_tensor = self.visualizer.read_image(hr_img_path)

        _, hr_img_height, hr_img_width = hr_img_tensor.shape

        if not lr_img_path:
            with open(config_paths[0], "r", encoding="UTF-8") as f:
                scaling_factor = yaml.safe_load(f)["model_params"]["scaling_factor"]

            lr_img_tensor = imresize_tensor(hr_img_tensor, scale=1 / scaling_factor, antialiasing=True)
        else:
            lr_img_tensor = self.visualizer.read_image(lr_img_path)

        results = {
            "Bicubic": F.interpolate(
                lr_img_tensor.unsqueeze(0),
                size=(hr_img_height, hr_img_width),
                mode="bicubic",
            )
            .squeeze(0)
            .clamp(0, 1)
        }

        for config_path in config_paths:
            evaluator = self._init_evaluator(config_path)
            results[config_path.parent.name] = evaluator.upscale(lr_img_tensor)

            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

        results["Original (HR)"] = hr_img_tensor

        if crop_size:
            crop_center_coords = self.visualizer.get_crop_center(img_tensor=hr_img_tensor, crop_size=crop_size)

            self.visualizer.create_collage_with_crop(
                hr_img_tensor=hr_img_tensor,
                sr_img_tensors_dict=results,
                crop_center=crop_center_coords,
                crop_size=crop_size,
                output_img_path=output_img_path,
            )
        else:
            self.visualizer.create_collage(
                sr_img_tensors_dict=results,
                output_img_path=output_img_path,
            )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    benchmark = Benchmark(device=device)

    models_to_test = list(Path("models").glob("*/config.yaml"))
    datasets = [Path("data/Set5"), Path("data/Set14")]

    benchmark.evaluate(
        config_paths=models_to_test,
        dataset_paths=datasets,
        save_csv_path=Path("results/final_benchmark.csv"),
    )

    benchmark.upscale(
        config_path=Path("models/SRResNet/config.yaml"),
        input_img_path=Path("images/hr_img_1.jpg"),
        output_img_path=Path("results/hat_hr_img_1.png"),
        downscale=True,
    )

    benchmark.compare(
        config_paths=models_to_test,
        hr_img_path=Path("images/img_073.png"),
        lr_img_path=Path("data/Urban100/LR_x4/img_073.png"),
        output_img_path=Path("results/img_073_comparison.png"),
        crop_size=64,
    )

    benchmark.compare(
        config_paths=models_to_test,
        hr_img_path=Path("images/hr_baboon.png"),
        output_img_path=Path("results/baboon_comparison.png"),
    )
