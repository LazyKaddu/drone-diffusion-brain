# models/unet1d.py

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding for scalar diffusion timesteps.
    Maps discrete timestep t in [0, T] to a continuous frequency vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [Batch] or [Batch, 1]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # [Batch, dim]


class Conv1dBlock(nn.Module):
    """
    Core temporal convolution block: Conv1d -> GroupNorm -> Mish.
    Preserves sequence length using symmetric padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    1D Temporal ResNet Block with Feature-wise Linear Modulation (FiLM).
    
    Architecture:
        x -> Conv1dBlock -> FiLM modulation -> Conv1dBlock -> + residual
                                   ^
                           cond ---| (scale gamma, shift beta)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(
            in_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
        )
        self.conv2 = Conv1dBlock(
            out_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
        )

        # FiLM Generator: projects combined condition vector to scale & shift factors
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        # Residual shortcut mapping if channel count differs
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Temporal action feature tensor [Batch, in_channels, Seq_Len]
            cond: Conditioning latent vector [Batch, cond_dim]
        Returns:
            Modulated feature tensor [Batch, out_channels, Seq_Len]
        """
        # First convolution
        h = self.conv1(x)

        # FiLM projection: split into multiplicative scale (gamma) and additive shift (beta)
        embed = self.cond_proj(cond)  # [Batch, out_channels * 2]
        embed = embed.unsqueeze(-1)   # [Batch, out_channels * 2, 1] for temporal broadcasting
        gamma, beta = torch.chunk(embed, 2, dim=1)

        # Apply affine modulation: (1 + gamma) * h + beta
        h = h * (1.0 + gamma) + beta

        # Second convolution
        h = self.conv2(h)

        # Add residual connection
        return h + self.residual_conv(x)


class Downsample1D(nn.Module):
    """Halves the temporal dimension using strided 1D convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Doubles the temporal dimension using transposed 1D convolution."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """
    1D Temporal U-Net for Diffusion Policies.
    
    Denoises continuous action trajectories [Batch, action_dim, pred_horizon]
    conditioned on physical environment state embeddings and diffusion timesteps.
    """

    def __init__(
        self,
        action_dim: int = 4,
        pred_horizon: int = 16,
        cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: List[int] = [128, 256, 512],
        kernel_size: int = 5,
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        # -------------------------------------------------------------
        # 1. Timestep and Conditioning Fusion
        # -------------------------------------------------------------
        self.time_emb = SinusoidalPosEmb(diffusion_step_embed_dim)
        
        # Total conditioning vector combines diffusion step + state observation latent
        total_cond_dim = diffusion_step_embed_dim + cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(total_cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        # -------------------------------------------------------------
        # 2. Input Stem (Projects 4 action channels into first feature dim)
        # -------------------------------------------------------------
        self.input_conv = nn.Conv1d(
            action_dim, down_dims[0], kernel_size=kernel_size, padding=kernel_size // 2
        )

        # -------------------------------------------------------------
        # 3. Encoder (Downsampling Path)
        # -------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_channels = down_dims[0]
        for i, out_channels in enumerate(down_dims):
            self.down_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        in_channels, out_channels, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
                    ),
                    ConditionalResidualBlock1D(
                        out_channels, out_channels, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
                    ),
                ])
            )
            # Downsample on all stages except the last
            if i < len(down_dims) - 1:
                self.downsamples.append(Downsample1D(out_channels))
            else:
                self.downsamples.append(nn.Identity())
            in_channels = out_channels

        # -------------------------------------------------------------
        # 4. Bottleneck (Mid Stage)
        # -------------------------------------------------------------
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(
            mid_dim, mid_dim, cond_dim=cond_dim,
            kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
        )
        self.mid_block2 = ConditionalResidualBlock1D(
            mid_dim, mid_dim, cond_dim=cond_dim,
            kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
        )

        # -------------------------------------------------------------
        # 5. Decoder (Upsampling Path with Skip Connections)
        # -------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_down_dims = list(reversed(down_dims))
        for i in range(len(reversed_down_dims) - 1):
            curr_dim = reversed_down_dims[i]
            next_dim = reversed_down_dims[i + 1]

            self.upsamples.append(Upsample1D(curr_dim, next_dim))
            # Cat with skip connection doubles input channels
            self.up_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        next_dim * 2, next_dim, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
                    ),
                    ConditionalResidualBlock1D(
                        next_dim, next_dim, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups, dropout=dropout
                    ),
                ])
            )

        # -------------------------------------------------------------
        # 6. Output Head (Maps back to 4 action channels)
        # -------------------------------------------------------------
        self.final_block = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(down_dims[0], action_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts noise residual added to action chunk.

        Args:
            sample: Noisy action sequence [Batch, action_dim (4), pred_horizon (16)]
            timestep: Diffusion step index [Batch] or scalar int
            cond: Environment conditioning latent vector [Batch, cond_dim (256)]

        Returns:
            Predicted noise tensor [Batch, action_dim (4), pred_horizon (16)]
        """
        # Ensure timestep is a 1D tensor
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        # Expand scalar timestep across batch if necessary
        if timestep.shape[0] != sample.shape[0]:
            timestep = timestep.repeat(sample.shape[0])

        # 1. Synthesize global condition vector (time + environment state)
        time_features = self.time_emb(timestep)
        global_cond = self.cond_mlp(torch.cat([time_features, cond], dim=-1))

        # 2. Input stem
        x = self.input_conv(sample)

        # 3. Downsampling path with skip saving
        skips = []
        for (res1, res2), downsample in zip(self.down_blocks, self.downsamples):
            x = res1(x, global_cond)
            x = res2(x, global_cond)
            skips.append(x)
            x = downsample(x)

        # 4. Bottleneck
        x = self.mid_block1(x, global_cond)
        x = self.mid_block2(x, global_cond)

        # 5. Upsampling path with skip concatenation
        for upsample, (res1, res2) in zip(self.upsamples, self.up_blocks):
            skip = skips.pop()
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = res1(x, global_cond)
            x = res2(x, global_cond)

        # 6. Final projection
        return self.final_block(x)


if __name__ == "__main__":
    # Smoke test for tensor shape invariance
    B, C, L = 8, 4, 16
    cond_dim = 256

    model = ConditionalUnet1D(
        action_dim=C,
        pred_horizon=L,
        cond_dim=cond_dim,
        down_dims=[128, 256, 512],
    )

    x_dummy = torch.randn(B, C, L)
    t_dummy = torch.randint(0, 100, (B,))
    cond_dummy = torch.randn(B, cond_dim)

    out = model(x_dummy, t_dummy, cond_dummy)
    print(f"Input shape:  {x_dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x_dummy.shape, f"Shape mismatch: {out.shape} != {x_dummy.shape}"
    print("Tensor shape preservation test passed successfully.")