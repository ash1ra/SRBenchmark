import math
from typing import Literal

from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_layer: bool = False,
        activation: Literal["prelu", "leaky_relu", "tanh"] | None = None,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential()

        self.conv_block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )

        if norm_layer:
            self.conv_block.append(nn.BatchNorm2d(out_channels))

        if activation:
            match activation.lower():
                case "prelu":
                    self.conv_block.append(nn.PReLU())
                case "leaky_relu":
                    self.conv_block.append(nn.LeakyReLU(0.2))
                case "tanh":
                    self.conv_block.append(nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class SubPixelConvBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.subpixel_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count * (scaling_factor**2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.subpixel_conv_block(x)


class ResBlock(nn.Module):
    def __init__(self, channels_count: int, kernel_size: int) -> None:
        super().__init__()

        self.res_block = nn.Sequential(
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                norm_layer=True,
                activation="prelu",
            ),
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                norm_layer=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.res_block(x) + x


class SRGAN(nn.Module):
    def __init__(
        self,
        channels_count: int,
        large_kernel_size: int,
        small_kernel_size: int,
        res_blocks_count: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=channels_count,
            kernel_size=large_kernel_size,
            activation="prelu",
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(channels_count=channels_count, kernel_size=small_kernel_size) for _ in range(res_blocks_count)]
        )

        self.conv_block2 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=small_kernel_size,
            norm_layer=True,
        )

        self.subpixel_conv_blocks = nn.Sequential(
            *[
                SubPixelConvBlock(
                    channels_count=channels_count,
                    kernel_size=small_kernel_size,
                    scaling_factor=2,
                )
                for _ in range(int(math.log2(scaling_factor)))
            ]
        )

        self.conv_block3 = ConvBlock(
            in_channels=channels_count,
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
