from typing import Literal

from torch import Tensor, nn


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels_count: int, reduction: int) -> None:
        super().__init__()

        self.avg_pool_layer = nn.AdaptiveAvgPool2d(output_size=1)
        self.layers_sequence = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count // reduction,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels_count // reduction,
                out_channels=channels_count,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.layers_sequence(self.avg_pool_layer(x))


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels_count: int, kernel_size: int, reduction: int) -> None:
        super().__init__()

        self.padding: int = kernel_size // 2

        self.layers_sequence = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
            ChannelAttentionBlock(channels_count=channels_count, reduction=reduction),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_sequence(x) + x


class ResidualGroup(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        reduction: int,
        rcab_count: int,
    ) -> None:
        super().__init__()

        self.layers_sequence = nn.Sequential()

        for _ in range(rcab_count):
            self.layers_sequence.append(
                ResidualChannelAttentionBlock(
                    channels_count=channels_count,
                    kernel_size=kernel_size,
                    reduction=reduction,
                )
            )

        self.layers_sequence.append(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_sequence(x) + x


class ResidualInResidualBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        reduction: int,
        rg_count: int,
        rcab_count: int,
    ) -> None:
        super().__init__()

        self.layers_sequence = nn.Sequential()

        for _ in range(rg_count):
            self.layers_sequence.append(
                ResidualGroup(
                    channels_count=channels_count,
                    kernel_size=kernel_size,
                    reduction=reduction,
                    rcab_count=rcab_count,
                )
            )

        self.layers_sequence.append(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_sequence(x) + x


class RCAN(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        reduction: int,
        rg_count: int,
        rcab_count: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.padding: int = kernel_size // 2

        self.head = nn.Conv2d(
            in_channels=3,
            out_channels=channels_count,
            kernel_size=kernel_size,
            padding=self.padding,
        )

        self.body = ResidualInResidualBlock(
            channels_count=channels_count,
            kernel_size=kernel_size,
            reduction=reduction,
            rg_count=rg_count,
            rcab_count=rcab_count,
        )

        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count * (scaling_factor**2),
                kernel_size=kernel_size,
                padding=self.padding,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=3,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_head = self.head(x)

        return self.tail(self.body(x_head) + x_head)
