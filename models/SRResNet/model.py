import math
from typing import Literal

from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_layer: bool = False,
        activation: Literal["prelu", "tanh"] | None = None,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential()

        self.conv_block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        if norm_layer:
            self.conv_block.append(nn.BatchNorm2d(out_channels))

        if activation:
            match activation.lower():
                case "prelu":
                    self.conv_block.append(nn.PReLU())
                case "tanh":
                    self.conv_block.append(nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class SubPixelConvBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.subpixel_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels * (scaling_factor**2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.subpixel_conv_block(x)


class ResBlock(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.res_block = nn.Sequential(
            ConvBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                norm_layer=True,
                activation="prelu",
            ),
            ConvBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                norm_layer=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.res_block(x) + x


class SRResNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        large_kernel_size: int,
        small_kernel_size: int,
        n_res_blocks: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            activation="prelu",
        )

        self.res_blocks = nn.Sequential(
            *[
                ResBlock(n_channels=n_channels, kernel_size=small_kernel_size)
                for _ in range(n_res_blocks)
            ]
        )

        self.conv_block2 = ConvBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            norm_layer=True,
        )

        self.subpixel_conv_blocks = nn.Sequential(
            *[
                SubPixelConvBlock(
                    n_channels=n_channels,
                    kernel_size=small_kernel_size,
                    scaling_factor=2,
                )
                for _ in range(int(math.log2(scaling_factor)))
            ]
        )

        self.conv_block3 = ConvBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            activation="tanh",
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv_block1(x)
        residual = output
        output = self.res_blocks(output)
        output = self.conv_block2(output)
        output += residual
        output = self.subpixel_conv_blocks(output)
        output = self.conv_block3(output)

        return output
