import torch
import torch.nn.functional as F
from torch import nn


class GlobLN(nn.Module):
    """Global Layer Normalization (globLN)."""

    def __init__(self, channel_size: int):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """Applies forward pass.
        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())
    
    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=groups,
        )
        self.norm = GlobLN(out_channels)
        self.act = nn.PReLU()

    def forward(self, input: torch.Tensor):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        d: int = 1,
        groups: int = 1,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=d,
            padding=((kernel_size - 1) // 2) * d,
            groups=groups,
        )
        self.norm = GlobLN(out_channels)

    def forward(self, input: torch.Tensor):
        output = self.conv(input)
        return self.norm(output)


class AFRCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 128,
        states: int = 4,
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            out_channels,
            in_channels,
            1,
            stride=1,
            groups=1,
        )
        self.depth = states
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels,
                in_channels,
                kernel_size=5,
                stride=1,
                groups=in_channels,
                d=1,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, states):
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kernel_size=5,
                    stride=2,
                    groups=in_channels,
                    d=1,
                )
            )
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(states):
            fuse_layer = nn.ModuleList([])
            for j in range(states):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        DilatedConvNorm(
                            in_channels,
                            in_channels,
                            kernel_size=5,
                            stride=2,
                            groups=in_channels,
                            d=1,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(states):
            if i == 0 or i == states - 1:
                self.concat_layer.append(
                    ConvNormAct(in_channels * 2, in_channels, 1, 1)
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(in_channels * 3, in_channels, 1, 1)
                )

        self.last_layer = nn.Sequential(
            ConvNormAct(in_channels * states, in_channels, 1, 1)
        )
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # # ----------parameters-------------
        # self.depth = states # Already defined!

    def forward(self, x: torch.Tensor):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(output1.device),
                    output[i],
                    F.interpolate(output[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(output1.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual