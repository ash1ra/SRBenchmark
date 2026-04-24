import math
from os import environ
from pathlib import Path
from typing import Any, TypedDict

environ["QT_LOGGING_RULES"] = "*=false"
environ["QT_QPA_PLATFORM"] = "xcb"
environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.io import ImageReadMode, read_image, write_png


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
    def save_image(self, img_tensor: Tensor, save_path: Path) -> None:
        img_tensor = img_tensor.cpu().clamp(0, 1)
        img_tensor = (img_tensor * 255.0).to(torch.uint8)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        write_png(img_tensor, str(save_path))

    def read_image(self, img_path: Path) -> Tensor:
        img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
        return img_tensor.float() / 255.0

    def print_results_table(self, model_name: str, results: dict[str, dict[str, float]]) -> None:
        print(f"{'=' * 82}")
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

    def save_benchmark_csv(
        self,
        results: dict[str, dict[str, dict[str, float]]],
        output_img_path: Path | str,
    ) -> None:
        flat_data = []
        for model_name, datasets in results.items():
            for dataset_name, metrics in datasets.items():
                row = {"Model": model_name, "Dataset": dataset_name}
                row.update(metrics)  # type: ignore
                flat_data.append(row)

        df = pd.DataFrame(flat_data)
        Path(output_img_path).parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_img_path, index=False, float_format="%.4f")
        print(f"✅ Table with results saved to {output_img_path}")

    def _draw_crop_preview(self) -> None:
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

        del self.crop_selection_params

        return result

    def create_collage_with_crop(
        self,
        hr_img_tensor: Tensor,
        sr_img_tensors_dict: dict[str, Tensor],
        crop_center: tuple[int, int],
        crop_size: int,
        output_img_path: Path,
    ) -> None:
        to_pil = transforms.ToPILImage()

        hr_img = to_pil(hr_img_tensor.cpu().clamp(0, 1))
        draw_hr = ImageDraw.Draw(hr_img)

        center_x, center_y = crop_center
        half_crop_size = crop_size // 2
        x1, y1 = center_x - half_crop_size, center_y - half_crop_size
        x2, y2 = center_x + half_crop_size, center_y + half_crop_size

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
        print(f"Collage saved to {output_img_path}")

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
        print(f"Comparison collage saved to {output_img_path}")
