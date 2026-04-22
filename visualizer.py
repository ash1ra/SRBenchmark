from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torchvision.io import ImageReadMode, read_image, write_png


class Visualizer:
    def __init__(self) -> None: ...

    @staticmethod
    def save_image(img_tensor: Tensor, save_path: Path) -> None:
        img_tensor = img_tensor.cpu().clamp(0, 1)
        img_tensor = (img_tensor * 255.0).to(torch.uint8)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        write_png(img_tensor, str(save_path))

    @staticmethod
    def read_image(img_path: Path) -> Tensor:
        img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
        return img_tensor.float() / 255.0

    @staticmethod
    def print_results_table(model_name: str, results: dict[str, dict[str, float]]) -> None:
        print(f"\n{'=' * 82}")
        print(f"{'Benchmark: ' + model_name:^82}")
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

    @staticmethod
    def save_benchmark_csv(
        results: dict[str, dict[str, dict[str, float]]],
        output_path: Path | str,
    ) -> None:
        flat_data = []
        for model_name, datasets in results.items():
            for dataset_name, metrics in datasets.items():
                row = {"Model": model_name, "Dataset": dataset_name}
                row.update(metrics)
                flat_data.append(row)

        df = pd.DataFrame(flat_data)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"✅ Table with results saved to {output_path}")
