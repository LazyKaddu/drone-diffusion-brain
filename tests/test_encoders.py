# tests/test_encoders.py

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.encoder import (
    KinematicEncoder,
    VisualEncoder,
    MultiModalEncoder,
    ObservationEncoder,
)


class TestKinematicEncoder:
    def test_forward_3d_tensor(self):
        batch_size = 4
        obs_horizon = 1
        obs_dim = 23
        latent_dim = 256

        encoder = KinematicEncoder(
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            latent_dim=latent_dim,
            hidden_dims=[128, 256],
        )

        x = torch.randn(batch_size, obs_horizon, obs_dim)
        out = encoder(x)
        assert out.shape == (batch_size, latent_dim)

    def test_forward_2d_tensor_expansion(self):
        batch_size = 4
        obs_dim = 23
        latent_dim = 128

        encoder = KinematicEncoder(
            obs_dim=obs_dim,
            obs_horizon=1,
            latent_dim=latent_dim,
        )

        x = torch.randn(batch_size, obs_dim)
        out = encoder(x)
        assert out.shape == (batch_size, latent_dim)

    def test_forward_multi_horizon(self):
        batch_size = 2
        obs_horizon = 3
        obs_dim = 23
        latent_dim = 256

        encoder = KinematicEncoder(
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            latent_dim=latent_dim,
        )

        x = torch.randn(batch_size, obs_horizon, obs_dim)
        out = encoder(x)
        assert out.shape == (batch_size, latent_dim)


class TestVisualEncoder:
    def test_forward_and_gn_replacement(self):
        batch_size = 2
        obs_horizon = 1
        latent_dim = 128

        encoder = VisualEncoder(
            obs_horizon=obs_horizon,
            latent_dim=latent_dim,
            pretrained=False,
            replace_bn_with_gn=True,
        )

        # Confirm no BatchNorm2d layers remain
        for m in encoder.modules():
            assert not isinstance(m, nn.BatchNorm2d)

        # Forward pass with [B, To, C, H, W]
        x = torch.randn(batch_size, obs_horizon, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (batch_size, latent_dim)

    def test_forward_4d_tensor_expansion(self):
        batch_size = 2
        latent_dim = 128
        encoder = VisualEncoder(
            obs_horizon=1,
            latent_dim=latent_dim,
            pretrained=False,
        )

        x = torch.randn(batch_size, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (batch_size, latent_dim)


class TestMultiModalEncoder:
    def test_forward(self):
        batch_size = 2
        obs_horizon = 1
        kin_dim = 23
        latent_dim = 256

        encoder = MultiModalEncoder(
            kin_dim=kin_dim,
            obs_horizon=obs_horizon,
            latent_dim=latent_dim,
            visual_latent_dim=64,
            kin_latent_dim=64,
        )

        obs = {
            "rgb": torch.randn(batch_size, obs_horizon, 3, 224, 224),
            "kin": torch.randn(batch_size, obs_horizon, kin_dim),
        }
        out = encoder(obs)
        assert out.shape == (batch_size, latent_dim)


class TestObservationEncoderFactory:
    def test_mlp_factory(self, mock_cfg):
        mock_cfg.policy.encoder.type = "mlp"
        encoder = ObservationEncoder(mock_cfg)
        assert isinstance(encoder.encoder, KinematicEncoder)

        x = torch.randn(2, mock_cfg.policy.obs_horizon, mock_cfg.policy.obs_dim)
        out = encoder(x)
        assert out.shape == (2, mock_cfg.policy.encoder.latent_dim)

    def test_visual_factory(self, mock_cfg):
        mock_cfg.policy.encoder.type = "resnet18"
        encoder = ObservationEncoder(mock_cfg)
        assert isinstance(encoder.encoder, VisualEncoder)

        x = torch.randn(2, mock_cfg.policy.obs_horizon, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, mock_cfg.policy.encoder.latent_dim)

    def test_multimodal_factory(self, mock_cfg):
        mock_cfg.policy.encoder.type = "multimodal"
        encoder = ObservationEncoder(mock_cfg)
        assert isinstance(encoder.encoder, MultiModalEncoder)

        obs = {
            "rgb": torch.randn(2, mock_cfg.policy.obs_horizon, 3, 224, 224),
            "kin": torch.randn(2, mock_cfg.policy.obs_horizon, mock_cfg.policy.obs_dim),
        }
        out = encoder(obs)
        assert out.shape == (2, mock_cfg.policy.encoder.latent_dim)

    def test_invalid_type_raises(self, mock_cfg):
        mock_cfg.policy.encoder.type = "unsupported_backbone"
        with pytest.raises(ValueError):
            ObservationEncoder(mock_cfg)
