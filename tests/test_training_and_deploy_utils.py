import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train import EMAModel, build_cosine_warmup_scheduler, to_device
from deploy import RealTimeNormalizer, format_kin_obs


class TestTrainingUtilities:
    def test_ema_model_step(self):
        model = nn.Linear(4, 4, bias=False)
        nn.init.constant_(model.weight, 1.0)

        ema = EMAModel(model, decay=0.9)
        assert torch.allclose(ema.averaged_model.weight, torch.tensor(1.0))

        # Change model weights to 2.0 and take EMA step
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.step(model)

        # Expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected_weight = 0.9 * 1.0 + 0.1 * 2.0
        assert torch.allclose(ema.averaged_model.weight, torch.tensor(expected_weight))

    def test_ema_state_dict_save_and_load(self):
        model = nn.Linear(2, 2)
        ema = EMAModel(model, decay=0.95)

        state = ema.state_dict()
        assert "weight" in state

        new_model = nn.Linear(2, 2)
        new_ema = EMAModel(new_model, decay=0.95)
        new_ema.load_state_dict(state)

        assert torch.allclose(new_ema.averaged_model.weight, ema.averaged_model.weight)

    def test_cosine_warmup_scheduler(self):
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=1.0)
        scheduler = build_cosine_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        # At step 0: lr should be 0.0
        assert math.isclose(scheduler.get_last_lr()[0], 0.0, abs_tol=1e-5)

        # Step 5: midway through warmup
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert math.isclose(scheduler.get_last_lr()[0], 0.5, abs_tol=1e-4)

        # Step 10: end of warmup, lr should be 1.0
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert math.isclose(scheduler.get_last_lr()[0], 1.0, abs_tol=1e-4)

        # Step 100: end of total steps, lr decays to near 0.0
        for _ in range(90):
            optimizer.step()
            scheduler.step()
        assert scheduler.get_last_lr()[0] < 0.05

    def test_to_device_recursive(self):
        device = torch.device("cpu")
        tensor = torch.tensor([1.0, 2.0])
        nested = {
            "a": tensor,
            "sub": {
                "b": torch.tensor([3.0, 4.0]),
                "c": "non_tensor_metadata",
            },
        }

        out = to_device(nested, device)
        assert isinstance(out, dict)
        assert out["a"].device == device
        assert out["sub"]["b"].device == device
        assert out["sub"]["c"] == "non_tensor_metadata"


class TestDeploymentUtilities:
    def test_real_time_normalizer_round_trip(self):
        stats = {
            "obs": {
                "min": [-10.0, 0.0],
                "range": [20.0, 10.0],
            },
            "action": {
                "min": [-1.0, 0.0],
                "range": [2.0, 4.0],
            },
        }
        device = torch.device("cpu")
        normalizer = RealTimeNormalizer(stats, device)

        # Test Observation Normalization: [-10.0, 0.0] -> [-1.0, -1.0]
        obs_min = torch.tensor([[-10.0, 0.0]], dtype=torch.float32)
        norm_min = normalizer.normalize_obs(obs_min)
        assert torch.allclose(norm_min, torch.tensor([[-1.0, -1.0]]))

        # Test Action Unnormalization: [1.0, 1.0] -> [1.0, 4.0]
        act_max_norm = np.array([[1.0, 1.0]], dtype=np.float32)
        phys_act = normalizer.unnormalize_action(act_max_norm)
        assert np.allclose(phys_act, np.array([[1.0, 4.0]], dtype=np.float32))

    def test_format_kin_obs_goal_concatenation(self):
        kin_20 = np.arange(20, dtype=np.float32)
        target_pos = np.array([1.5, -2.5, 3.0], dtype=np.float32)

        # When expected obs_dim is 23:
        formatted_23 = format_kin_obs(kin_20, obs_dim=23, target_pos=target_pos)
        assert formatted_23.shape == (23,)
        assert np.allclose(formatted_23[:20], kin_20)
        assert np.allclose(formatted_23[20:], target_pos)

        # When expected obs_dim is 20:
        formatted_20 = format_kin_obs(kin_20, obs_dim=20, target_pos=target_pos)
        assert formatted_20.shape == (20,)
        assert np.allclose(formatted_20, kin_20)
