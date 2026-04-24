from pathlib import Path
from typing import Literal, Optional, overload

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch.nn import functional as F


def get_available_models(models_dir: Path) -> list[str]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    models = []
    for item in models_dir.iterdir():
        if item.is_dir() and (item / "config.yaml").exists():
            models.append(item.name)

    return sorted(models)


def rgb2y(img_tensor: Tensor) -> Tensor:
    weights = torch.tensor(
        [65.481, 128.553, 24.966],
        device=img_tensor.device,
        dtype=img_tensor.dtype,
    )
    bias = 16.0

    return (einsum(img_tensor, weights, "b c h w, c -> b h w").unsqueeze(1) + bias) / 255


def calculate_psnr(sr_img_tensor: Tensor, hr_img_tensor: Tensor, crop_border: int) -> float:
    if sr_img_tensor.dim() == 3:
        sr_img_tensor = rearrange(sr_img_tensor, "c h w -> 1 c h w")
        hr_img_tensor = rearrange(hr_img_tensor, "c h w -> 1 c h w")

    sr_img_tensor = torch.round(rgb2y(sr_img_tensor) * 255.0)
    hr_img_tensor = torch.round(rgb2y(hr_img_tensor) * 255.0)

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((sr_img_tensor - hr_img_tensor) ** 2)

    if mse == 0:
        return float("inf")
    else:
        return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()


def create_window_for_ssim_metric(window_size: int, sigma: float, num_channels: int) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    window_1d = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma**2))
    window_1d /= window_1d.sum()

    window_2d = einsum(window_1d, window_1d, "i, j -> i j")

    return repeat(window_2d, "h w -> c 1 h w", c=num_channels)


@overload
def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    return_map: Literal[False],
    window: Optional[Tensor] = None,
) -> float: ...


@overload
def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    return_map: Literal[True],
    window: Optional[Tensor] = None,
) -> tuple[float, Tensor]: ...


def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    window: Optional[Tensor] = None,
    return_map: bool = False,
) -> float | tuple[float, Tensor]:
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    if window is None:
        window = create_window_for_ssim_metric(
            window_size=window_size,
            sigma=1.5,
            num_channels=sr_img_tensor.size(1),
        ).to(device=sr_img_tensor.device, dtype=sr_img_tensor.dtype)

    mu_input = torch.cat([sr_img_tensor, hr_img_tensor], dim=0)
    mu_output = F.conv2d(
        input=mu_input,
        weight=window,
        stride=1,
        padding=0,
        groups=sr_img_tensor.size(1),
    )

    mu1, mu2 = mu_output.chunk(2, dim=0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma_input = torch.cat(
        [
            sr_img_tensor * sr_img_tensor,
            hr_img_tensor * hr_img_tensor,
            sr_img_tensor * hr_img_tensor,
        ],
        dim=0,
    )
    sigma_output = F.conv2d(
        input=sigma_input,
        weight=window,
        stride=1,
        padding=0,
        groups=sr_img_tensor.size(1),
    )

    conv_sr_sq, conv_hr_sq, conv_sr_hr = sigma_output.chunk(3, dim=0)

    sigma1_sq = conv_sr_sq - mu1_sq
    sigma2_sq = conv_hr_sq - mu2_sq
    sigma12 = conv_sr_hr - mu1_mu2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    mean_metric = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    ssim_map = mean_metric * contrast_metric
    ssim_value = ssim_map.mean().item()

    if return_map:
        return ssim_value, ssim_map
    else:
        return ssim_value


@overload
def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: Literal[False] = False,
) -> float: ...


@overload
def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: Literal[True] = True,
) -> tuple[float, Tensor]: ...


def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: bool = False,
) -> float | tuple[float, Tensor | None]:
    if sr_img_tensor.dim() == 3:
        sr_img_tensor = rearrange(sr_img_tensor, "c h w -> 1 c h w")
        hr_img_tensor = rearrange(hr_img_tensor, "c h w -> 1 c h w")

    sr_img_tensor = rgb2y(sr_img_tensor) * 255.0
    hr_img_tensor = rgb2y(hr_img_tensor) * 255.0

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    if sr_img_tensor.size(-1) < window_size or sr_img_tensor.size(-2) < window_size:
        if return_map:
            return 0.0, None
        else:
            return 0.0

    return ssim_metric(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        window_size=window_size,
        return_map=return_map,
    )


# Ported from BasicSR (matlab_functions.py) to ensure academic reproducibility.
# Implements MATLAB-like bicubic interpolation required for standard SR benchmarks (Set5, Set14).
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py


def cubic(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3

    return (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * ((abs_x <= 1).astype(type(abs_x))) + (
        -0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2
    ) * (((abs_x > 1) * (abs_x <= 2)).astype(type(abs_x)))


def calculate_weights_indices(
    in_length: int,
    out_length: int,
    scaling_factor: float,
    kernel_width: int,
    antialiasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if (scaling_factor < 1) and antialiasing:
        kernel_width: int | float = kernel_width / scaling_factor

    x = np.linspace(1, out_length, out_length)
    u = x / scaling_factor + 0.5 * (1 - 1 / scaling_factor)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2

    indices = left.reshape(int(out_length), 1) + np.linspace(0, p - 1, p).reshape(1, int(p))

    distance_to_center = u.reshape(int(out_length), 1) - indices

    if (scaling_factor < 1) and antialiasing:
        weights = scaling_factor * cubic(distance_to_center * scaling_factor)
    else:
        weights = cubic(distance_to_center)

    weights_sum = np.sum(weights, 1).reshape(int(out_length), 1)
    weights /= weights_sum

    weights_zero_idx = np.where(weights_sum == 0)
    if len(weights_zero_idx[0]) > 0:
        weights[weights_zero_idx, 0] = 1

    padded_indices = indices.astype(int)
    padded_indices -= 1

    padded_indices = np.abs(padded_indices)
    padded_indices = np.where(padded_indices < in_length, padded_indices, 2 * in_length - 1 - padded_indices)
    padded_indices = np.clip(padded_indices, 0, in_length - 1)

    return weights, padded_indices


def imresize(img_tensor: Tensor, scaling_factor: float, antialiasing: bool = True) -> Tensor:
    if scaling_factor == 1:
        return img_tensor

    img_np = img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    if len(img_np.shape) == 3:
        input_img_height, input_img_width, input_img_num_channels = img_np.shape
    else:
        input_img_height, input_img_width = img_np.shape
        input_img_num_channels = 1

    output_img_height = int(np.ceil(input_img_height * scaling_factor))
    output_img_width = int(np.ceil(input_img_width * scaling_factor))

    kernel_width = 4

    height_weights, height_indices = calculate_weights_indices(
        in_length=input_img_height,
        out_length=output_img_height,
        scaling_factor=scaling_factor,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    width_weights, width_indices = calculate_weights_indices(
        in_length=input_img_width,
        out_length=output_img_width,
        scaling_factor=scaling_factor,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    img_aug = np.zeros((output_img_height, input_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img_np[:, :, channel] if input_img_num_channels > 1 else img_np
        pixels = channel_data[height_indices]
        img_aug[:, :, channel] = np.sum(height_weights[:, :, None] * pixels, axis=1)

    output_img = np.zeros((output_img_height, output_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img_aug[:, :, channel]
        pixels = channel_data[:, width_indices]
        output_img[:, :, channel] = np.sum(width_weights[None, :, :] * pixels, axis=2)

    output_img *= 255.0
    output_img = np.round(np.clip(output_img, 0.0, 255.0)).astype(np.uint8)
    output_img = output_img.astype(np.float32) / 255.0

    return torch.from_numpy(output_img.copy()).permute(2, 0, 1).float()
