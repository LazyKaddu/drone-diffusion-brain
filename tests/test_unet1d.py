# tests/test_unet1d.py

import pytest
import torch

from models.unet1d import (
    SinusoidalPosEmb,
    Conv1dBlock,
    ConditionalResidualBlock1D,
    Downsample1D,
    Upsample1D,
    ConditionalUnet1D,
)


class TestSinusoidalPosEmb:
    def test_pos_emb_shape_and_bounds(self):
        dim = 128
        pos_emb = SinusoidalPosEmb(dim=dim)

        # 1D tensor
        t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
        out = pos_emb(t)
        assert out.shape == (4, dim)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

        # 2D tensor
        t_2d = torch.tensor([[5], [20]], dtype=torch.long)
        out_2d = pos_emb(t_2d)
        assert out_2d.shape == (2, dim)


class TestResidualAndSamplingBlocks:
    def test_conv1d_block_preserves_length(self):
        block = Conv1dBlock(in_channels=16, out_channels=32, kernel_size=5, n_groups=8)
        x = torch.randn(4, 16, 16)
        out = block(x)
        assert out.shape == (4, 32, 16)

    def test_conditional_residual_block_1d(self):
        res_block = ConditionalResidualBlock1D(
            in_channels=32,
            out_channels=64,
            cond_dim=128,
            kernel_size=5,
            n_groups=8,
        )
        x = torch.randn(4, 32, 16)
        cond = torch.randn(4, 128)
        out = res_block(x, cond)
        assert out.shape == (4, 64, 16)

    def test_downsample_and_upsample(self):
        down = Downsample1D(dim=32)
        up = Upsample1D(in_dim=32, out_dim=32)

        x = torch.randn(2, 32, 16)
        x_down = down(x)
        assert x_down.shape == (2, 32, 8)

        x_up = up(x_down)
        assert x_up.shape == (2, 32, 16)


class TestConditionalUnet1D:
    def test_temporal_shape_preservation(self):
        B, C, L = 4, 4, 16
        cond_dim = 256
        model = ConditionalUnet1D(
            action_dim=C,
            pred_horizon=L,
            cond_dim=cond_dim,
            down_dims=[64, 128, 256],
            kernel_size=5,
            n_groups=8,
        )

        sample = torch.randn(B, C, L)
        timestep = torch.randint(0, 100, (B,))
        cond = torch.randn(B, cond_dim)

        out = model(sample, timestep, cond)
        assert out.shape == (B, C, L)

    def test_gradient_flow_and_no_nans(self):
        B, C, L = 2, 4, 16
        cond_dim = 128
        model = ConditionalUnet1D(
            action_dim=C,
            pred_horizon=L,
            cond_dim=cond_dim,
            down_dims=[32, 64],
            kernel_size=5,
            n_groups=4,
        )

        sample = torch.randn(B, C, L, requires_grad=True)
        timestep = torch.tensor([10, 20])
        cond = torch.randn(B, cond_dim, requires_grad=True)

        out = model(sample, timestep, cond)
        loss = out.sum()
        loss.backward()

        assert sample.grad is not None
        assert not torch.isnan(sample.grad).any()
        assert cond.grad is not None
        assert not torch.isnan(cond.grad).any()

        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None grad"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
