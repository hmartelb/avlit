import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.modules.autoencoder import FrameAutoEncoder
from src.modules.afrcnn import AFRCNN, GlobLN


class AVLIT(nn.Module):
    def __init__(
        self,
        num_sources: int = 2,
        # Audio branch
        kernel_size: int = 40,
        audio_hidden_channels: int = 512,
        audio_bottleneck_channels: int = 128,
        audio_num_blocks: int = 8,
        audio_states: int = 5,
        # Video branch
        video_hidden_channels: int = 128,
        video_bottleneck_channels: int = 128,
        video_num_blocks: int = 4,
        video_states: int = 5,
        video_encoder_checkpoint: str = "",
        video_encoder_trainable: bool = False,
        video_embedding_dim: int = 1024,
        # AV fusion
        fusion_operation: str = "sum",
        fusion_positions: list[int] = [4],
    ) -> None:
        super().__init__()

        self.num_sources = num_sources
        self.kernel_size = kernel_size
        self.audio_states = audio_states
        self.audio_hidden_channels = audio_hidden_channels
        self.video_embedding_dim = video_embedding_dim

        # Audio encoder
        self.audio_encoder = nn.Conv1d(
            in_channels=1,
            out_channels=audio_hidden_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.audio_encoder.weight)

        # Audio decoder
        self.audio_decoder = nn.ConvTranspose1d(
            in_channels=audio_hidden_channels,
            out_channels=1,
            output_padding=kernel_size // 2 - 1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.audio_decoder.weight)

        # Video encoder
        self.video_encoder = FrameAutoEncoder()
        if os.path.isfile(video_encoder_checkpoint):
            self.video_encoder.load_state_dict(torch.load(video_encoder_checkpoint))
        if not video_encoder_trainable:
            for p in self.video_encoder.parameters():
                p.requires_grad = False

        # Audio adaptation
        self.audio_norm = GlobLN(audio_hidden_channels)
        self.audio_bottleneck = nn.Conv1d(
            in_channels=audio_hidden_channels,
            out_channels=audio_bottleneck_channels,
            kernel_size=1,
        )

        # Video adaptation
        self.video_bottleneck = nn.Conv1d(
            in_channels=num_sources * video_embedding_dim,
            out_channels=video_bottleneck_channels,
            kernel_size=1,
        )

        # Masking
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=audio_bottleneck_channels,
                out_channels=num_sources * audio_hidden_channels,
                kernel_size=1,
            ),
        )
        self.mask_activation = nn.ReLU()

        # Audio branch
        self.audio_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=audio_hidden_channels,
            bottleneck_channels=audio_bottleneck_channels,
            num_blocks=audio_num_blocks,
            states=audio_states,
            fusion_operation=fusion_operation,
            fusion_positions=fusion_positions,
        )

        # Video branch
        self.video_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=video_hidden_channels,
            bottleneck_channels=video_bottleneck_channels,
            num_blocks=video_num_blocks,
            states=video_states,
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor):
        # Get sizes of inputs
        b, T = x.shape[0], x.shape[-1]
        M, F = v.shape[1], v.shape[2]

        # Get audio features, fa
        x = self._pad_input(x)
        fa_in = self.audio_encoder(x)
        fa = self.audio_norm(fa_in)
        fa = self.audio_bottleneck(fa)

        # Get video features, fv
        fv = self.video_encoder.encode(v)
        fv = fv.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
        fv = self.video_bottleneck(fv)

        # Forward the video and audio branches
        fv_p = self.video_branch(fv)
        fa_p = self.audio_branch(fa, fv_p)

        # Apply masking
        fa_m = self._masking(fa_in, fa_p)

        # Decode audio
        fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
        s = self.audio_decoder(fa_m)
        s = s.view(b, self.num_sources, -1)
        s = self._trim_output(s, T)
        return s

    def _masking(self, f, m):
        m = self.mask_net(m)
        m = m.view(
            m.shape[0],
            self.num_sources,
            self.audio_hidden_channels,
            -1,
        )
        m = self.mask_activation(m)
        masked = m * f.unsqueeze(1)
        return masked

    def lcm(self):
        half_kernel = self.kernel_size // 2
        pow_states = 2**self.audio_states
        return abs(half_kernel * pow_states) // math.gcd(half_kernel, pow_states)

    def _pad_input(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm()
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1])
                + [appropriate_shape[-1] + self.lcm() - values_to_pad],
                dtype=torch.float32,
            ).to(x.device)
            padded_x[..., : x.shape[-1]] = x
            return padded_x
        return x

    def _trim_output(self, x, T):
        if x.shape[-1] >= T:
            return x[..., 0:T]
        return x


class IterativeBranch(nn.Module):
    def __init__(
        self,
        num_sources: int = 2,
        hidden_channels: int = 512,
        bottleneck_channels: int = 128,
        num_blocks: int = 8,
        states: int = 5,
        fusion_operation: str = "sum",
        fusion_positions: list = [0],
    ) -> None:
        super().__init__()

        # Branch attributes
        self.num_sources = num_sources
        self.hidden_channels = hidden_channels
        self.bottleneck_channels = bottleneck_channels
        self.num_blocks = num_blocks
        self.states = states
        self.fusion_operation = fusion_operation
        assert fusion_operation in [
            "sum",
            "prod",
            "concat",
        ], f"The specified fusion_operation is not supported, must be one of ['sum', 'prod', 'concat']."
        self.fusion_positions = list(
            filter(lambda x: x < num_blocks and x >= 0, fusion_positions)
        )
        assert (
            len(fusion_positions) > 0
        ), f"The length of the fusion positions must be non-zero. Make sure to specify values between 1 and num_blocks ({num_blocks})"

        # Modules
        self.afrcnn_block = AFRCNN(
            in_channels=hidden_channels,
            out_channels=bottleneck_channels,
            states=states,
        )
        self.adapt_audio = nn.Sequential(
            nn.Conv1d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                groups=bottleneck_channels,
            ),
            nn.PReLU(),
        )
        if len(self.fusion_positions) > 0:
            self.adapt_fusion = nn.Sequential(
                nn.Conv1d(
                    bottleneck_channels * (2 if fusion_operation == "concat" else 1),
                    bottleneck_channels,
                    kernel_size=1,
                    stride=1,
                    groups=bottleneck_channels,
                ),
                nn.PReLU(),
            )

    def forward(
        self,
        fa: torch.Tensor,
        fv_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i in range(self.num_blocks):
            # 1) Get the input: base case fa, else last output + fa
            Ri = fa if i == 0 else self.adapt_audio(Ri + fa)

            # 2) Apply modality fusion ?
            if i in self.fusion_positions and fv_p is not None:
                f = self._modality_fusion(Ri, fv_p)
                Ri = self.adapt_fusion(f)

            # 3) Apply the A-FRCNN block
            Ri = self.afrcnn_block(Ri)
        return Ri

    def _modality_fusion(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] > b.shape[-1]:
            b = F.interpolate(b, size=a.shape[2:])

        if self.fusion_operation == "sum":
            return a + b
        elif self.fusion_operation == "prod":
            return a * b
        elif self.fusion_operation == "concat":
            return torch.cat([a, b], dim=1)
        return a
