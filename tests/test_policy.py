# tests/test_policy.py

import pytest
import torch

from models.policy import DroneDiffusionPolicy


class TestDroneDiffusionPolicy:
    def test_policy_initialization(self, mock_cfg):
        policy = DroneDiffusionPolicy(mock_cfg)
        assert policy.action_dim == 4
        assert policy.pred_horizon == 16
        assert policy.exec_horizon == 4
        assert policy.obs_horizon == 1

    def test_policy_compute_loss(self, mock_cfg, dummy_batch):
        policy = DroneDiffusionPolicy(mock_cfg)
        loss = policy.compute_loss(dummy_batch)

        assert torch.is_tensor(loss)
        assert loss.ndim == 0  # Scalar loss
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_policy_predict_action_batched(self, mock_cfg, dummy_batch):
        policy = DroneDiffusionPolicy(mock_cfg)
        policy.eval()

        obs = dummy_batch["obs"]
        batch_size = obs.shape[0]

        with torch.no_grad():
            actions = policy.predict_action(obs, num_inference_steps=2)

        assert actions.shape == (batch_size, mock_cfg.policy.pred_horizon, mock_cfg.policy.action_dim)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)

    def test_policy_predict_action_unbatched_2d_tensor(self, mock_cfg):
        policy = DroneDiffusionPolicy(mock_cfg)
        policy.eval()

        # Input unbatched observation of shape [To, obs_dim]
        obs_2d = torch.randn(mock_cfg.policy.obs_horizon, mock_cfg.policy.obs_dim)

        with torch.no_grad():
            actions = policy.predict_action(obs_2d, num_inference_steps=2)

        assert actions.shape == (1, mock_cfg.policy.pred_horizon, mock_cfg.policy.action_dim)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)

    def test_policy_get_receding_actions(self, mock_cfg, dummy_batch):
        policy = DroneDiffusionPolicy(mock_cfg)
        policy.eval()

        obs = dummy_batch["obs"]
        batch_size = obs.shape[0]

        with torch.no_grad():
            receding_actions = policy.get_receding_actions(obs, exec_horizon=4)

        assert receding_actions.shape == (batch_size, 4, mock_cfg.policy.action_dim)
