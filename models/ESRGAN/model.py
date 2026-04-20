import math
from typing import Literal

import torch
import torch.nn.init as init
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19


def _init_scaled_weights(module: nn.Module, scale: float):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight.data, a=0.0, mode="fan_in")
        module.weight.data *= scale

        if module.bias is not None:
            init.constant_(module.bias.data, 0.0)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        activation: Literal["leaky_relu", "tanh"] | None = None,
        bias: bool = True,
        norm_layer: bool = False,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential()

        self.conv_block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        if norm_layer:
            self.conv_block.append(nn.BatchNorm2d(out_channels))

        if activation:
            match activation.lower():
                case "leaky_relu":
                    self.conv_block.append(
                        nn.LeakyReLU(
                            negative_slope=0.2,
                            inplace=True,
                        )
                    )
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
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count * (scaling_factor**2),
                kernel_size=kernel_size,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.subpixel_conv_block(x)


class ResDenseBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        kernel_size: int,
        conv_layers_count: int,
    ) -> None:
        super().__init__()

        self.conv_layers_count: int = conv_layers_count
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(
            ConvBlock(
                in_channels=channels_count,
                out_channels=growth_channels_count,
                kernel_size=kernel_size,
                activation="leaky_relu",
            )
        )

        for i in range(1, conv_layers_count - 1):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=channels_count + i * growth_channels_count,
                    out_channels=growth_channels_count,
                    kernel_size=kernel_size,
                    activation="leaky_relu",
                )
            )

        self.conv_layers.append(
            ConvBlock(
                in_channels=channels_count + (conv_layers_count - 1) * growth_channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs = [x]
        outputs.append(self.conv_layers[0](x))

        for i in range(1, self.conv_layers_count):
            outputs.append(self.conv_layers[i](torch.cat(outputs, 1)))

        return outputs[-1] * 0.2 + x


class RRDB(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        kernel_size: int,
        conv_layers_count: int,
        res_dense_blocks_count: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            *[
                ResDenseBlock(
                    channels_count=channels_count,
                    growth_channels_count=growth_channels_count,
                    kernel_size=kernel_size,
                    conv_layers_count=conv_layers_count,
                )
                for _ in range(res_dense_blocks_count)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) * 0.2 + x


class ESRGAN(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        large_kernel_size: int,
        small_kernel_size: int,
        res_dense_blocks_count: int,
        rrdb_count: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=channels_count,
            kernel_size=large_kernel_size,
            padding=4,
        )

        self.rrdb = nn.Sequential(
            *[
                RRDB(
                    channels_count=channels_count,
                    growth_channels_count=growth_channels_count,
                    kernel_size=small_kernel_size,
                    conv_layers_count=5,
                    res_dense_blocks_count=res_dense_blocks_count,
                )
                for _ in range(rrdb_count)
            ]
        )

        self.conv_block2 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=small_kernel_size,
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
            padding=4,
            activation="tanh",
        )

        self.apply(lambda fn: _init_scaled_weights(fn, scale=0.1))

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv_block1(x)
        residual = output
        output = self.rrdb(output)
        output = self.conv_block2(output)
        output += residual
        output = self.subpixel_conv_blocks(output)
        output = self.conv_block3(output)

        return output
