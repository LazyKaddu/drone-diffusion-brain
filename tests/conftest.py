import os
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure repository root containing train.py and deploy.py is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.zarr_dataset import ZarrDataWriter


@pytest.fixture
def mock_cfg():
    """Provides a deterministic, lightweight Hydra/OmegaConf configuration for testing on CPU."""
    cfg = OmegaConf.create({
        "project": {
            "name": "drone-diffusion-test",
            "seed": 42,
            "device": "cpu",
        },
        "env": {
            "id": "CustomDrone-v0",
            "obs_type": "kin",
            "physics_hz": 240,
            "control_hz": 20,
            "gui": False,
        },
        "data": {
            "zarr_path": "data/mock_trajectories.zarr",
            "num_episodes": 10,
            "max_episode_steps": 100,
            "val_split": 0.25,
            "num_workers": 0,
            "pin_memory": False,
        },
        "policy": {
            "obs_horizon": 1,
            "pred_horizon": 16,
            "exec_horizon": 4,
            "action_dim": 4,
            "obs_dim": 23,
            "encoder": {
                "type": "mlp",
                "latent_dim": 256,
                "hidden_dims": [128, 256],
                "activation": "Mish",
            },
            "unet": {
                "down_dims": [128, 256, 512],
                "kernel_size": 5,
                "n_groups": 8,
                "cond_dim": 256,
                "diffusion_step_embed_dim": 128,
                "dropout": 0.0,
            },
            "diffusion": {
                "train_scheduler": {
                    "type": "DDPMScheduler",
                    "num_train_timesteps": 100,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                    "beta_schedule": "squaredcos_cap_v2",
                    "clip_sample": True,
                    "prediction_type": "epsilon",
                },
                "inference_scheduler": {
                    "type": "DDIMScheduler",
                    "num_inference_timesteps": 10,
                    "eta": 0.0,
                },
            },
        },
        "normalization": {
            "action": {
                "min": [-1.0, -1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0, 1.0],
            },
            "obs": {
                "clip_min": -50.0,
                "clip_max": 50.0,
            },
        },
        "training": {
            "batch_size": 16,
            "num_epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "lr_warmup_steps": 10,
            "grad_clip_norm": 1.0,
            "use_ema": True,
            "ema_decay": 0.99,
            "save_interval": 1,
            "checkpoint_dir": "checkpoints_test",
            "resume_checkpoint": None,
        },
    })
    return cfg


@pytest.fixture
def temp_zarr_store(tmp_path):
    """
    Creates an isolated synthetic 3-episode Zarr v2 archive on disk using ZarrDataWriter.
    Automatically cleaned up by pytest upon test teardown.
    """
    zarr_dir = str(tmp_path / "synthetic_trajectories.zarr")
    obs_dim = 23
    action_dim = 4
    img_shape = (3, 224, 224)

    writer = ZarrDataWriter(
        zarr_path=zarr_dir,
        obs_dim=obs_dim,
        action_dim=action_dim,
        img_shape=img_shape,
        img_chunks=50,
        data_chunks=100,
        overwrite=True,
    )

    ep_lengths = [30, 45, 25]
    for i, ep_len in enumerate(ep_lengths):
        ep_img = np.random.randint(0, 255, size=(ep_len, *img_shape), dtype=np.uint8)
        # 20 kinematics + 3 goal
        kin = np.random.randn(ep_len, 20).astype(np.float32)
        goal = np.tile(np.array([1.0, -1.0, 0.5], dtype=np.float32), (ep_len, 1))
        ep_obs = np.hstack([kin, goal]).astype(np.float32)
        ep_act = np.random.uniform(-1.0, 1.0, size=(ep_len, action_dim)).astype(np.float32)

        writer.append_episode(ep_img, ep_obs, ep_act)

    writer.close()
    return {
        "path": zarr_dir,
        "ep_lengths": ep_lengths,
        "total_steps": sum(ep_lengths),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "img_shape": img_shape,
    }


@pytest.fixture
def dummy_batch():
    """Provides a standard batch of observations and actions."""
    batch_size = 4
    obs_horizon = 1
    pred_horizon = 16
    obs_dim = 23
    action_dim = 4

    return {
        "obs": torch.randn(batch_size, obs_horizon, obs_dim),
        "action": torch.randn(batch_size, pred_horizon, action_dim),
    }
