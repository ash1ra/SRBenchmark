import gc
from pathlib import Path
from typing import Literal

import torch
from torch.nn import functional as F

from core_utils import ModelConfig, imresize
from evaluator import Evaluator
from logger import logger
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
        output_dir: Path | None = None,
    ) -> None:
        results = {}

        for config_path in config_paths:
            evaluator = self._init_evaluator(config_path)
            model_name = evaluator.model_name

            logger.info(f"[{model_name}] | Starting benchmark...")

            model_results = evaluator.evaluate(dataset_paths)
            results[model_name] = model_results

            stats = evaluator.get_model_stats((3, 64, 64))

            for dataset_name in model_results:
                model_results[dataset_name]["Params (M)"] = stats["params"] / 1e6  # type: ignore
                model_results[dataset_name]["FLOPs (G)"] = stats["macs"] / 1e9 * 2  # type: ignore

            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

        self.visualizer.print_comparative_table(results)

        if output_dir and results:
            self.visualizer.generate_plots(results, output_dir)
            self.visualizer.save_benchmark_csv(results, output_dir)

    def upscale(
        self,
        config_paths: list[Path],
        input_img_paths: list[Path],
        output_img_paths: list[Path],
        downscale: bool = False,
    ) -> None:
        for config_path in config_paths:
            evaluator = self._init_evaluator(config_path)

            for input_img_path, output_img_path in zip(input_img_paths, output_img_paths):
                img_tensor = self.visualizer.read_image(input_img_path)

                if downscale:
                    sr_img_tensor = evaluator.upscale_downscaled(img_tensor)
                else:
                    sr_img_tensor = evaluator.upscale(img_tensor)

                output_img_path = output_img_path.parent / f"{evaluator.model_name.lower()}_{output_img_path.name}.png"

                self.visualizer.save_image(sr_img_tensor, output_img_path)

            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

    def compare(
        self,
        config_paths: list[Path],
        hr_img_paths: list[Path],
        output_img_paths: list[Path],
        lr_img_paths: list[Path | None] | None = None,
        crop_size: int | tuple[int, int] | None = None,
    ) -> None:
        scaling_factors = [ModelConfig.from_yaml(path).scaling_factor for path in config_paths]

        if len(set(scaling_factors)) > 1:
            raise ValueError(
                f"Cannot compare models with different scaling factors in a single run! "
                f"Found factors: {scaling_factors}"
            )

        scaling_factor = scaling_factors[0]

        hr_img_tensors = [self.visualizer.read_image(path) for path in hr_img_paths]
        lr_img_tensors = []

        for i, hr_img_tensor in enumerate(hr_img_tensors):
            lr_img_path = lr_img_paths[i] if lr_img_paths else None
            if lr_img_path is None:
                lr_img_tensor = imresize(hr_img_tensor, scaling_factor=1 / scaling_factor, antialiasing=True)
                lr_img_tensors.append(lr_img_tensor)
            else:
                lr_img_tensor = self.visualizer.read_image(lr_img_path)
                lr_img_tensors.append(lr_img_tensor)

        results = []
        for hr_img_tensor, lr_img_tensor in zip(hr_img_tensors, lr_img_tensors):
            _, hr_img_height, hr_img_width = hr_img_tensor.shape
            results.append(
                {
                    "Bicubic": F.interpolate(
                        lr_img_tensor.unsqueeze(0),
                        size=(hr_img_height, hr_img_width),
                        mode="bicubic",
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                }
            )

        for config_path in config_paths:
            evaluator = self._init_evaluator(config_path)
            model_name = config_path.parent.name

            for i, lr_img_tensor in enumerate(lr_img_tensors):
                results[i][model_name] = evaluator.upscale(lr_img_tensor)

            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

        for i, hr_img_tensor in enumerate(hr_img_tensors):
            results[i]["Original (HR)"] = hr_img_tensor

            if crop_size:
                crop_center_coords = self.visualizer.get_crop_center(img_tensor=hr_img_tensor, crop_size=crop_size)

                self.visualizer.create_collage_with_crop(
                    hr_img_tensor=hr_img_tensor,
                    sr_img_tensors_dict=results[i],
                    crop_center=crop_center_coords,
                    crop_size=crop_size,
                    output_img_path=output_img_paths[i],
                )
            else:
                self.visualizer.create_collage(
                    sr_img_tensors_dict=results[i],
                    output_img_path=output_img_paths[i],
                )
