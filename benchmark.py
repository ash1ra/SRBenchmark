import gc
from pathlib import Path
from typing import Literal

import torch

from evaluator import Evaluator
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
