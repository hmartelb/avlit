from torch import nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        leaky_slope: float = 0.3,
    ):
        """
        Encoder block module that performs convolution, instance normalization, and leaky ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple): Stride of the convolution operation.
            leaky_slope (float): Negative slope coefficient for the leaky ReLU activation.

        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after convolution, instance normalization, and activation.

        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        leaky_slope: float = 0.3,
    ):
        """
        Decoder block module that performs transposed convolution, instance normalization, and leaky ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the transposed convolution kernel.
            stride (int or tuple): Stride of the transposed convolution operation.
            leaky_slope (float): Negative slope coefficient for the leaky ReLU activation.

        """
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EncoderAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cout = base_channels * (2**i)
            cin = in_channels if i == 0 else cout // 2
            self.layers.append(EncoderBlock(cin, cout, 2, 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cin = base_channels * (2 ** (num_layers - i - 1))
            cout = in_channels if i == num_layers - 1 else cin // 2
            self.layers.append(DecoderBlock(cin, cout, 2, 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FrameAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        num_layers: int = 3,
    ):
        """
        Single-frame autoencoder used to obtain frame-level embeddings from a video. 

        Args:
            - in_channels (int, optional): Number of video channels (1: grayscale, 3: rgb, ...). Defaults to 1.
            - base_channels (int, optional): Number of channels in the first convolutional layer, multiplied by 2 each subsequent layer. Defaults to 8.
            - num_layers (int, optional): Number of layers (i.e. depth of the autoencoder). Defaults to 3.
        """
        super().__init__()
        self.encoder = EncoderAE(in_channels, base_channels, num_layers)
        self.decoder = DecoderAE(in_channels, base_channels, num_layers)

    def forward(self, x):
        return self.reconstruct(x)

    def encode(self, x):
        # x is expected to be a tensor of shape [batch, num_sources, frames, w, h].
        # Convert it to [batch * num_sources * frames, w, h]
        batch, num_sources, frames, w, h = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        x = x.contiguous().view(batch * num_sources * frames, 1, w, h)

        z = self.encoder(x)

        # Undo the view of x. z has [batch * num_sources * frames, c', w', h']
        # Convert it to [batch, num_sources, frames, c' * w' * h']
        z = z.view(batch, num_sources, frames, -1)
        return z

    def reconstruct(self, x):
        # x is expected to be a tensor of shape [batch, num_sources, frames, w, h].
        # Convert it to [batch * num_sources * frames, w, h]
        batch, num_sources, frames, w, h = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        x = x.contiguous().view(batch * num_sources * frames, 1, w, h)

        z = self.encoder(x)
        y = self.decoder(z)

        # Undo the view of x. y has [batch * num_sources * frames, w, h]
        # Convert it to [batch, num_sources, frames, w, h]
        y = y.view(batch, num_sources, frames, w, h)
        return y
