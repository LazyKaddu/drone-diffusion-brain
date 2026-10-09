# models/encoder.py

import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torchvision import models
from omegaconf import OmegaConf

_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")
_default_cfg = OmegaConf.load(_config_path)


class KinematicEncoder(nn.Module):
    """
    MLP encoder for low-dimensional proprioceptive states (IMU, velocities, positions).
    Flattens temporal observation horizon [B, To, obs_dim] into a fixed latent conditioning vector.
    """

    def __init__(
        self,
        obs_dim: int = _default_cfg.policy.obs_dim,
        obs_horizon: int = _default_cfg.policy.obs_horizon,
        latent_dim: int = _default_cfg.policy.encoder.latent_dim,
        hidden_dims: List[int] = _default_cfg.policy.encoder.hidden_dims,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.latent_dim = latent_dim

        in_dim = obs_dim * obs_horizon
        layers = []

        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Mish(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            ])
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, latent_dim))
        layers.append(nn.LayerNorm(latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Kinematics tensor of shape [B, To, obs_dim] or [B, obs_dim]
        Returns:
            Latent conditioning vector [B, latent_dim]
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)  # Expand to [B, 1, obs_dim]

        batch_size = obs.shape[0]
        flattened = obs.reshape(batch_size, -1)  # [B, To * obs_dim]
        return self.net(flattened)


class VisualEncoder(nn.Module):
    """
    ResNet-18 visual encoder for ego-camera observations.
    Replaces global average pooling head with a low-dimensional FiLM projection.
    Replaces BatchNorm with GroupNorm for small batch sizes typical of robotics policies.
    """

    def __init__(
        self,
        obs_horizon: int = _default_cfg.policy.obs_horizon,
        latent_dim: int = _default_cfg.policy.encoder.latent_dim,
        pretrained: bool = False,
        replace_bn_with_gn: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.latent_dim = latent_dim

        # Load ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        if replace_bn_with_gn:
            backbone = self._replace_submodules(
                root_module=backbone,
                predicate=lambda m: isinstance(m, nn.BatchNorm2d),
                func=lambda m: nn.GroupNorm(num_groups=m.num_features // 16 or 1, num_channels=m.num_features),
            )

        # Handle multi-frame stacking across channels if fed concatenated,
        # or process frames independently and pool
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Extract feature extractor without standard classification FC
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,  # Matches replaced GN (it was already modified in-place)
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        # Output feature dimension of ResNet-18 is 512
        feat_dim = 512 * obs_horizon
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _replace_submodules(self, root_module: nn.Module, predicate, func) -> nn.Module:
        """Recursively swaps BatchNorm layers with GroupNorm."""
        for name, child in root_module.named_children():
            if predicate(child):
                setattr(root_module, name, func(child))
            else:
                self._replace_submodules(child, predicate, func)
        return root_module

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: Image tensor [B, To, C, H, W] or [B, C, H, W]
        Returns:
            Latent conditioning vector [B, latent_dim]
        """
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)  # [B, 1, C, H, W]

        b, to, c, h, w = rgb.shape
        # Fold temporal horizon into batch dimension for parallel forward pass
        x = rgb.view(b * to, c, h, w)
        features = self.backbone(x)  # [B * To, 512, 1, 1]
        features = features.view(b, to * 512)
        return self.proj(features)


class MultiModalEncoder(nn.Module):
    """
    Fuses visual camera frames and kinematic telemetry into a unified conditioning latent.
    """

    def __init__(
        self,
        kin_dim: int = _default_cfg.policy.obs_dim,
        obs_horizon: int = _default_cfg.policy.obs_horizon,
        latent_dim: int = _default_cfg.policy.encoder.latent_dim,
        visual_latent_dim: int = 128,
        kin_latent_dim: int = 128,
    ):
        super().__init__()
        self.visual_net = VisualEncoder(
            obs_horizon=obs_horizon,
            latent_dim=visual_latent_dim,
        )
        self.kin_net = KinematicEncoder(
            obs_dim=kin_dim,
            obs_horizon=obs_horizon,
            latent_dim=kin_latent_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(visual_latent_dim + kin_latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: Dict containing 'rgb' and 'kin' tensors.
        """
        vis_emb = self.visual_net(obs["rgb"])
        kin_emb = self.kin_net(obs["kin"])
        fused = torch.cat([vis_emb, kin_emb], dim=-1)
        return self.fusion(fused)


class ObservationEncoder(nn.Module):
    """
    Factory interface selecting between kinematic, visual, or multimodal architectures
    using the pipeline configuration dictionary.
    """

    def __init__(self, cfg):
        super().__init__()
        encoder_type = cfg.policy.encoder.type.lower()
        latent_dim = cfg.policy.encoder.latent_dim
        obs_horizon = cfg.policy.obs_horizon

        if encoder_type == "mlp":
            self.encoder = KinematicEncoder(
                obs_dim=cfg.policy.obs_dim,
                obs_horizon=obs_horizon,
                latent_dim=latent_dim,
                hidden_dims=list(cfg.policy.encoder.hidden_dims),
            )
        elif encoder_type in ["resnet18", "vision", "rgb"]:
            self.encoder = VisualEncoder(
                obs_horizon=obs_horizon,
                latent_dim=latent_dim,
            )
        elif encoder_type == "multimodal":
            self.encoder = MultiModalEncoder(
                kin_dim=cfg.policy.obs_dim,
                obs_horizon=obs_horizon,
                latent_dim=latent_dim,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, obs: Union[torch.Tensor, dict]) -> torch.Tensor:
        return self.encoder(obs)


if __name__ == "__main__":
    # Smoke verification
    batch_size = 4
    obs_horizon = 1
    kin_dim = 20
    latent_dim = 256

    # 1. Test Kinematic Encoder
    kin_encoder = KinematicEncoder(obs_dim=kin_dim, obs_horizon=obs_horizon, latent_dim=latent_dim)
    dummy_kin = torch.randn(batch_size, obs_horizon, kin_dim)
    kin_out = kin_encoder(dummy_kin)
    print(f"Kinematic Encoder Out: {kin_out.shape}")
    assert kin_out.shape == (batch_size, latent_dim)

    # 2. Test Visual Encoder
    vis_encoder = VisualEncoder(obs_horizon=obs_horizon, latent_dim=latent_dim)
    dummy_rgb = torch.randn(batch_size, obs_horizon, 3, 224, 224)
    vis_out = vis_encoder(dummy_rgb)
    print(f"Visual Encoder Out:    {vis_out.shape}")
    assert vis_out.shape == (batch_size, latent_dim)
    print("All encoder checks passed.")