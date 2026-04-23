from typing import Literal, Optional, overload

import torch
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch.nn import functional as F

from prepare_data import imresize


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


def imresize_tensor(img_tensor: Tensor, scale: float, antialiasing: bool = True) -> Tensor:
    device = img_tensor.device

    img_np = img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    lr_np = imresize(img_np, scale=scale, antialiasing=antialiasing)
    lr_tensor = torch.from_numpy(lr_np.copy()).permute(2, 0, 1).float()

    return lr_tensor.to(device).clamp(0, 1)
