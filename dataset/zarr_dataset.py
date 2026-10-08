# dataset/zarr_dataset.py

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class DroneZarrDataset(Dataset):
    """
    High-throughput sliding-window trajectory dataset for Diffusion Policy.
    Reads chunked kinematic/vision observations and action sequences from a Zarr store.

    Expected Zarr structure:
        data/
            obs: (N, obs_dim) float32
            action: (N, action_dim) float32
        meta/
            episode_ends: (num_episodes,) int64
    """

    def __init__(
        self,
        zarr_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 1,
        exec_horizon: int = 4,
        stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ):
        """
        Args:
            zarr_path: Path to the .zarr root directory.
            pred_horizon (Ta): Action sequence length predicted by the U-Net.
            obs_horizon (To): Number of past observation frames fed into the condition encoder.
            exec_horizon (Te): Number of steps executed before replanning (stored for reference).
            stats: Precomputed normalizer min/max statistics. If None, computed over the dataset.
        """
        super().__init__()
        self.zarr_path = zarr_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.exec_horizon = exec_horizon

        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr directory not found at: {zarr_path}")

        # Open Zarr in read-only mode (thread-safe for multi-worker PyTorch DataLoaders)
        self.root = zarr.open(zarr_path, mode="r")

        # Handle either nested groups or root-level arrays
        if "data" in self.root:
            self.obs_arr = self.root["data/obs"]
            self.action_arr = self.root["data/action"]
            self.episode_ends = np.array(self.root["meta/episode_ends"][:], dtype=np.int64)
        else:
            self.obs_arr = self.root["obs"]
            self.action_arr = self.root["action"]
            self.episode_ends = np.array(self.root["episode_ends"][:], dtype=np.int64)

        self.total_steps = len(self.obs_arr)

        # Build index mapping: sample_idx -> (step_idx, ep_start, ep_end)
        self.indices = self._build_indices()

        # Compute or assign normalization statistics
        self.stats = stats if stats is not None else self._compute_stats()

    def _build_indices(self) -> np.ndarray:
        """
        Precomputes the lookup table for sequence slicing across episode boundaries.
        Each valid step index maps to its parent episode's start and end index.
        """
        indices = []
        ep_start = 0

        for ep_end in self.episode_ends:
            ep_len = ep_end - ep_start
            if ep_len <= 0:
                continue

            for step in range(ep_start, ep_end):
                indices.append([step, ep_start, ep_end])

            ep_start = ep_end

        return np.array(indices, dtype=np.int64)

    def _compute_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Computes min and max bounds for observations and actions across the dataset
        for tensor normalization into [-1.0, 1.0].
        """
        # Read arrays in memory-efficient chunks if large
        obs_data = self.obs_arr[:]
        action_data = self.action_arr[:]

        stats = {
            "obs": {
                "min": np.min(obs_data, axis=0).astype(np.float32),
                "max": np.max(obs_data, axis=0).astype(np.float32),
            },
            "action": {
                "min": np.min(action_data, axis=0).astype(np.float32),
                "max": np.max(action_data, axis=0).astype(np.float32),
            },
        }

        # Guard against zero-variance channels (division by zero)
        for key in ["obs", "action"]:
            diff = stats[key]["max"] - stats[key]["min"]
            diff[diff < 1e-4] = 1.0
            stats[key]["range"] = diff

        return stats

    @staticmethod
    def normalize(x: np.ndarray, stat_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Normalizes input array linearly to range [-1.0, 1.0]."""
        return 2.0 * (x - stat_dict["min"]) / stat_dict["range"] - 1.0

    @staticmethod
    def unnormalize(x: Union[np.ndarray, torch.Tensor], stat_dict: Dict[str, np.ndarray]) -> Union[np.ndarray, torch.Tensor]:
        """Inverts [-1.0, 1.0] normalization back to original physical units."""
        if isinstance(x, torch.Tensor):
            device = x.device
            min_val = torch.from_numpy(stat_dict["min"]).to(device)
            range_val = torch.from_numpy(stat_dict["range"]).to(device)
            return (x + 1.0) / 2.0 * range_val + min_val
        return (x + 1.0) / 2.0 * stat_dict["range"] + stat_dict["min"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extracts temporal sliding window:
        - Observation condition: [To, obs_dim]
        - Action prediction chunk: [Ta, action_dim]
        """
        step_idx, ep_start, ep_end = self.indices[idx]

        # -----------------------------------------------------------------
        # 1. Observation Horizon Extraction [step_idx - To + 1 : step_idx + 1]
        # -----------------------------------------------------------------
        obs_start = step_idx - self.obs_horizon + 1
        if obs_start < ep_start:
            # Replicate boundary frame for start-of-episode steps
            pad_len = ep_start - obs_start
            raw_obs = self.obs_arr[ep_start : step_idx + 1]
            pad_block = np.repeat(raw_obs[0:1], pad_len, axis=0)
            obs_seq = np.concatenate([pad_block, raw_obs], axis=0)
        else:
            obs_seq = self.obs_arr[obs_start : step_idx + 1]

        # -----------------------------------------------------------------
        # 2. Action Horizon Extraction [step_idx : step_idx + Ta]
        # -----------------------------------------------------------------
        act_end = step_idx + self.pred_horizon
        if act_end > ep_end:
            # Replicate last available action to pad remaining steps at episode tail
            raw_act = self.action_arr[step_idx:ep_end]
            pad_len = act_end - ep_end
            pad_block = np.repeat(raw_act[-1:], pad_len, axis=0)
            action_seq = np.concatenate([raw_act, pad_block], axis=0)
        else:
            action_seq = self.action_arr[step_idx:act_end]

        # -----------------------------------------------------------------
        # 3. Normalization into [-1.0, 1.0]
        # -----------------------------------------------------------------
        norm_obs = self.normalize(obs_seq, self.stats["obs"])
        norm_action = self.normalize(action_seq, self.stats["action"])

        return {
            "obs": torch.tensor(norm_obs, dtype=torch.float32),          # [To, obs_dim]
            "action": torch.tensor(norm_action, dtype=torch.float32),    # [Ta, action_dim]
        }


def get_dataloaders(cfg) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Constructs train and validation DataLoaders using train/val split ratios.
    """
    full_dataset = DroneZarrDataset(
        zarr_path=cfg.data.zarr_path,
        pred_horizon=cfg.policy.pred_horizon,
        obs_horizon=cfg.policy.obs_horizon,
        exec_horizon=cfg.policy.exec_horizon,
    )

    val_size = int(len(full_dataset) * cfg.data.val_split)
    train_size = len(full_dataset) - val_size

    # Split sequentially or deterministically by episode rather than shuffle-shattering trajectories
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.project.seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, full_dataset.stats