import math
from os import environ
from pathlib import Path
from typing import Any, TypedDict

environ["QT_LOGGING_RULES"] = "*=false"
environ["QT_QPA_PLATFORM"] = "xcb"
environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.io import ImageReadMode, read_image, write_png

from logger import logger


class CropSelectionParams(TypedDict):
    img_bgr: np.ndarray
    img_width: int
    img_height: int
    crop_width: int
    crop_height: int
    center_x: int
    center_y: int
    window_name: str


class Visualizer:
    def __init__(self) -> None:
        self.crop_selection_params: CropSelectionParams | None = None

    def save_image(self, img_tensor: Tensor, save_path: Path) -> None:
        img_tensor = img_tensor.cpu().clamp(0, 1)
        img_tensor = (img_tensor * 255.0).to(torch.uint8)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        write_png(img_tensor, str(save_path))

    def read_image(self, img_path: Path) -> Tensor:
        img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
        return img_tensor.float() / 255.0

    def print_comparative_table(self, results: dict[str, dict[str, dict[str, float]]]) -> None:
        models = list(results.keys())
        if not models:
            return
        datasets = list(results[models[0]].keys())

        metrics = ["Params (M)", "FLOPs (G)", "Time", "PSNR", "SSIM", "LPIPS", "CLIPIQA", "MUSIQ"]

        directions = {
            "Params (M)": False,
            "FLOPs (G)": False,
            "Time": False,
            "PSNR": True,
            "SSIM": True,
            "LPIPS": False,
            "CLIPIQA": True,
            "MUSIQ": True,
        }

        headers = [
            "Params (M) (↓)",
            "FLOPs (G) (↓)",
            "Time (↓)",
            "PSNR (↑)",
            "SSIM (↑)",
            "LPIPS (↓)",
            "CLIPIQA(↑)",
            "MUSIQ (↑)",
        ]

        table_width = 135
        table_str = f"\n{'=' * table_width}\n"
        table_str += f"{'Comparative Benchmark Results':^{table_width}}\n"
        table_str += f"{'=' * table_width}\n"

        for dataset in datasets:
            table_str += f"Dataset: {dataset}\n"
            table_str += f"{'-' * table_width}\n"

            header_str = f"{'Model':<15} |"
            for h in headers:
                header_str += f" {h:>11} |"
            table_str += header_str[:-1] + "\n"
            table_str += f"{'-' * table_width}\n"

            top_markers = {metric: {} for metric in metrics}
            for metric in metrics:
                model_vals = {model: results[model][dataset].get(metric, 0.0) for model in models}
                valid_vals = [v for v in model_vals.values() if v > 0.0] or [0.0]

                sorted_vals = sorted(list(set(valid_vals)), reverse=directions[metric])
                top1 = sorted_vals[0] if len(sorted_vals) > 0 else None
                top2 = sorted_vals[1] if len(sorted_vals) > 1 else None

                for model, val in model_vals.items():
                    if val == 0.0 and metric in ["Params (M)", "FLOPs (G)"]:
                        top_markers[metric][model] = ""
                    elif val == top1:
                        top_markers[metric][model] = "(1)"
                    elif val == top2:
                        top_markers[metric][model] = "(2)"
                    else:
                        top_markers[metric][model] = ""

            for model in models:
                row_str = f"{model:<15} |"
                for metric in metrics:
                    val = results[model][dataset].get(metric, 0.0)
                    marker = top_markers[metric][model]

                    if metric in ["Params (M)", "FLOPs (G)"]:
                        val_str = f"{val:.2f}"
                    elif metric == "Time":
                        val_str = f"{val:.3f}"
                    elif metric in ["PSNR", "MUSIQ"]:
                        val_str = f"{val:.2f}"
                    else:
                        val_str = f"{val:.4f}"

                    cell = f"{val_str}{marker}"
                    row_str += f" {cell:>11} |"
                table_str += row_str[:-1] + "\n"
            table_str += f"{'=' * table_width}\n"

        logger.info(table_str)

    def save_benchmark_csv(
        self,
        results: dict[str, dict[str, dict[str, float]]],
        output_img_path: Path,
    ) -> None:
        flat_data = []
        metrics_order = ["Params (M)", "FLOPs (G)", "Time", "PSNR", "SSIM", "LPIPS", "CLIPIQA", "MUSIQ"]

        for model_name, datasets in results.items():
            for dataset_name, metrics in datasets.items():
                row = {"Model": model_name, "Dataset": dataset_name}
                for metric in metrics_order:
                    if metric in metrics:
                        row[metric] = metrics[metric]  # type: ignore
                flat_data.append(row)

        Path(output_img_path).parent.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame(flat_data)
        df.to_csv(output_img_path / "benchmark.csv", index=False, float_format="%.4f")

        logger.info(f"Table with results saved to {output_img_path}")

    def _draw_crop_preview(self) -> None:
        if self.crop_selection_params is None:
            raise ValueError("self.crop_selection_params is None")

        img = self.crop_selection_params["img_bgr"].copy()

        center_x = self.crop_selection_params["center_x"]
        center_y = self.crop_selection_params["center_y"]
        crop_width = self.crop_selection_params["crop_width"]
        crop_height = self.crop_selection_params["crop_height"]

        x1 = center_x - crop_width // 2
        y1 = center_y - crop_height // 2
        x2 = center_x + crop_width // 2
        y2 = center_y + crop_height // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.circle(img, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=-1)

        cv2.imshow(self.crop_selection_params["window_name"], img)

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if self.crop_selection_params is None:
            raise ValueError("self.crop_selection_params is None")

        if event == cv2.EVENT_LBUTTONDOWN:
            crop_width = self.crop_selection_params["crop_width"]
            crop_height = self.crop_selection_params["crop_height"]
            img_width = self.crop_selection_params["img_width"]
            img_height = self.crop_selection_params["img_height"]

            self.crop_selection_params["center_x"] = max(crop_width // 2, min(x, img_width - crop_width // 2))
            self.crop_selection_params["center_y"] = max(crop_height // 2, min(y, img_height - crop_height // 2))

            self._draw_crop_preview()

    def get_crop_center(self, img_tensor: Tensor, crop_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(crop_size, int):
            crop_width, crop_height = crop_size, crop_size
        else:
            crop_width, crop_height = crop_size

        img_np = img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_bgr = cv2.cvtColor((img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_height, img_width = img_bgr.shape[:2]

        self.crop_selection_params: CropSelectionParams = {
            "img_bgr": img_bgr,
            "img_width": img_width,
            "img_height": img_height,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "center_x": img_width // 2,
            "center_y": img_height // 2,
            "window_name": "Select Crop Center (Click, then press ENTER or ESC)",
        }

        cv2.namedWindow(self.crop_selection_params["window_name"], cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.crop_selection_params["window_name"], self._mouse_callback)

        self._draw_crop_preview()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in [13, 27]:
                break

        cv2.destroyAllWindows()

        result = (self.crop_selection_params["center_x"], self.crop_selection_params["center_y"])

        self.crop_selection_params = None

        return result

    def create_collage_with_crop(
        self,
        hr_img_tensor: Tensor,
        sr_img_tensors_dict: dict[str, Tensor],
        crop_center: tuple[int, int],
        crop_size: int | tuple[int, int],
        output_img_path: Path,
    ) -> None:
        to_pil = transforms.ToPILImage()

        hr_img = to_pil(hr_img_tensor.cpu().clamp(0, 1))
        draw_hr = ImageDraw.Draw(hr_img)

        if isinstance(crop_size, int):
            crop_width, crop_height = crop_size, crop_size
        else:
            crop_width, crop_height = crop_size

        center_x, center_y = crop_center
        half_width = crop_width // 2
        half_height = crop_height // 2

        x1, y1 = center_x - half_width, center_y - half_height
        x2, y2 = center_x + half_width, center_y + half_height

        line_width = max(3, hr_img.width // 300)
        draw_hr.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

        crops = []
        labels = []
        for label, img_tensor in sr_img_tensors_dict.items():
            crops.append(to_pil(img_tensor[:, y1:y2, x1:x2].cpu().clamp(0, 1)))
            labels.append(label)

        full_img_height = min(600, hr_img.height)
        full_img_width = int(hr_img.width * full_img_height / hr_img.height)
        full_img_resized = hr_img.resize((full_img_width, full_img_height), Image.Resampling.LANCZOS)

        num_crops = len(crops)
        rows = 2
        columns = math.ceil(num_crops / rows)

        header_height = 40
        patch_size = (full_img_height // rows) - header_height

        resized_crops = [crop.resize((patch_size, patch_size), Image.Resampling.NEAREST) for crop in crops]

        padding = 20
        canvas_width = full_img_width + padding + (columns * patch_size) + ((columns - 1) * 10)
        canvas_height = full_img_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        canvas.paste(full_img_resized, (0, 0))

        grid_start_x = full_img_width + padding

        for i, (crop, label) in enumerate(zip(resized_crops, labels)):
            row = i // columns
            column = i % columns

            x_offset = grid_start_x + column * (patch_size + 10)
            y_offset = row * (patch_size + header_height)

            canvas.paste(crop, (x_offset, y_offset))

            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x_offset + (patch_size - text_width) // 2
            text_y = y_offset + patch_size + 5

            draw.text((text_x, text_y), label, fill="black", font=font)

        output_img_path.parent.mkdir(exist_ok=True, parents=True)
        canvas.save(output_img_path)
        logger.info(f"Collage saved to '{output_img_path}'")

    def create_collage(self, sr_img_tensors_dict: dict[str, Tensor], output_img_path: Path) -> None:
        imgs = []
        labels = []

        for label, tensor in sr_img_tensors_dict.items():
            imgs.append(transforms.ToPILImage()(tensor.cpu().clamp(0, 1)))
            labels.append(label)

        img_width, img_height = imgs[0].size
        header_height = 60
        num_imgs = len(imgs)

        if num_imgs > 2 and num_imgs % 2 == 0:
            rows = 2
            cols = num_imgs // 2
        else:
            rows = 1
            cols = num_imgs

        canvas_width = img_width * cols
        canvas_height = (img_height + header_height) * rows

        canvas = Image.new("RGB", (canvas_width, canvas_height), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 26)
        except OSError:
            font = ImageFont.load_default()

        for i, (img, label) in enumerate(zip(imgs, labels)):
            row = i // cols
            column = i % cols

            x_offset = column * img_width
            y_offset = row * (img_height + header_height)

            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x_offset + (img_width - text_width) // 2
            text_y = y_offset + img_height + (header_height - text_height) // 2 - 7

            canvas.paste(img, (x_offset, y_offset))
            draw.text((text_x, text_y), label, fill=(40, 40, 40), font=font)

        output_img_path.parent.mkdir(exist_ok=True, parents=True)
        canvas.save(output_img_path)
        logger.info(f"Comparison collage saved to '{output_img_path}'")

    def generate_plots(self, results: dict[str, dict[str, dict[str, float]]], output_dir: Path) -> None:
        models = list(results.keys())
        if not models:
            return

        datasets = list(results[models[0]].keys())
        output_dir.mkdir(exist_ok=True, parents=True)

        avg_metrics = {model: {} for model in models}
        metrics_list = ["PSNR", "SSIM", "LPIPS", "CLIPIQA", "MUSIQ", "Time", "Params (M)", "MACs (G)"]

        for model in models:
            for metric in metrics_list:
                vals = [
                    results[model][dataset].get(metric, 0.0)
                    for dataset in datasets
                    if results[model][dataset].get(metric, 0.0) > 0.0
                ]
                avg_metrics[model][metric] = sum(vals) / len(vals) if vals else 0.0

        colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(models)))

        self._plot_scatter_tradeoff(models, avg_metrics, colors, output_dir)
        self._plot_radar_balance(models, avg_metrics, colors, output_dir)
        self._plot_grouped_bar_stability(models, datasets, results, colors, output_dir)

    def _plot_scatter_tradeoff(
        self,
        models: list[str],
        avg_metrics: dict,
        colors: np.ndarray,
        output_dir: Path,
    ) -> None:
        plt.figure(figsize=(10, 6))

        for i, model in enumerate(models):
            x = avg_metrics[model].get("Time", 0.0)
            y = avg_metrics[model].get("PSNR", 0.0)
            s = avg_metrics[model].get("Params (M)", 0.0) * 15 + 50

            if x > 0 and y > 0:
                plt.scatter(x, y, s=s, color=colors[i], label=model, alpha=0.7, edgecolors="black", linewidth=1)
                plt.annotate(model, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10)

        plt.xlabel("Inference Time per Image (seconds) ↓", fontsize=12)
        plt.ylabel("Average PSNR ↑", fontsize=12)
        plt.title("Trade-off: Speed vs Quality\n(Bubble size represents Model Parameters in Millions)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = output_dir / "plot_1_scatter_tradeoff.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Scatter plot saved to '{plot_path}'")

    def _plot_radar_balance(self, models: list[str], avg_metrics: dict, colors: np.ndarray, output_dir: Path) -> None:
        radar_metrics = ["PSNR", "SSIM", "CLIPIQA", "MUSIQ", "LPIPS"]
        min_max = {}

        for metric in radar_metrics:
            vals = [avg_metrics[model][metric] for model in models if avg_metrics[model][metric] > 0]
            if vals:
                min_max[metric] = {"min": min(vals), "max": max(vals)}
            else:
                min_max[metric] = {"min": 0, "max": 1}

        angles = [n / float(len(radar_metrics)) * 2 * math.pi for n in range(len(radar_metrics))]
        angles += angles[:1]

        _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], radar_metrics, size=11)
        ax.set_yticklabels([])

        for i, model in enumerate(models):
            values = []
            for metric in radar_metrics:
                val = avg_metrics[model].get(metric, 0.0)
                mn, mx = min_max[metric]["min"], min_max[metric]["max"]

                norm_val = 1.0 if mx == mn else (val - mn) / (mx - mn)
                if metric == "LPIPS":
                    norm_val = 1.0 - norm_val

                values.append(0.1 + norm_val * 0.9)

            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=model, color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.1)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title("Model Perception-Distortion Balance\n(Further from center = Better)", size=14, y=1.1)

        plot_path = output_dir / "plot_2_radar_balance.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Radar chart saved to '{plot_path}'")

    def _plot_grouped_bar_stability(
        self,
        models: list[str],
        datasets: list[str],
        results: dict,
        colors: np.ndarray,
        output_dir: Path,
    ) -> None:
        x = np.arange(len(datasets))
        width = 0.8 / len(models)

        _, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            psnr_vals = [results[model][dataset].get("PSNR", 0.0) for dataset in datasets]
            offset = (i - len(models) / 2 + 0.5) * width
            rects = ax.bar(x + offset, psnr_vals, width, label=model, color=colors[i])
            ax.bar_label(rects, padding=3, fmt="%.1f", fontsize=9)

        ax.set_ylabel("PSNR (↑)", fontsize=12)
        ax.set_title("Quality Stability Across Datasets", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=len(models))

        all_psnr = [results[metric][dataset].get("PSNR", 0.0) for metric in models for dataset in datasets]
        valid_psnr = [v for v in all_psnr if v > 0]
        if valid_psnr:
            ax.set_ylim(bottom=max(0, min(valid_psnr) - 2), top=max(valid_psnr) + 2)

        plot_path = output_dir / "plot_3_bar_stability.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Grouped Bar chart saved to '{plot_path}'")
