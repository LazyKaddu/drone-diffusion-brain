# dataset/zarr_dataset.py

import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from numcodecs import Blosc
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# High-Throughput Zarr Trajectory Writer
# ---------------------------------------------------------------------------

class ZarrDataWriter:
    """
    High-throughput chunked Zarr v2 storage manager for drone flight trajectories.
    Writes camera frames, kinematic states, and action commands with compression.
    """

    def __init__(
        self,
        zarr_path: str,
        obs_dim: int = 23,
        action_dim: int = 4,
        img_shape: Tuple[int, int, int] = (3, 224, 224),
        img_chunks: int = 500,
        data_chunks: int = 1000,
        overwrite: bool = False,
    ):
        self.zarr_path = zarr_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.img_shape = img_shape
        self.img_chunks = img_chunks
        self.data_chunks = data_chunks

        if os.path.exists(zarr_path):
            if overwrite:
                shutil.rmtree(zarr_path)
            else:
                raise FileExistsError(f"Zarr store '{zarr_path}' already exists and overwrite=False.")

        self.root = zarr.open(self.zarr_path, mode="w")
        self.data_group = self.root.create_group("data")
        self.meta_group = self.root.create_group("meta")

        # High-performance compression configuration
        img_compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)
        vec_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

        # 1. Image frames [Total_Steps, 3, H, W]
        self.img_zarr = self.data_group.zeros(
            "img",
            shape=(0, *img_shape),
            chunks=(img_chunks, *img_shape),
            dtype="uint8",
            compressor=img_compressor,
        )

        # 2. Kinematic state observations [Total_Steps, obs_dim]
        self.obs_zarr = self.data_group.zeros(
            "obs",
            shape=(0, obs_dim),
            chunks=(data_chunks, obs_dim),
            dtype="float32",
            compressor=vec_compressor,
        )

        # 3. Flight control actions [Total_Steps, action_dim]
        self.act_zarr = self.data_group.zeros(
            "action",
            shape=(0, action_dim),
            chunks=(data_chunks, action_dim),
            dtype="float32",
            compressor=vec_compressor,
        )

        # 4. Episode boundaries metadata [Num_Episodes]
        self.ep_ends_zarr = self.meta_group.zeros(
            "episode_ends",
            shape=(0,),
            chunks=(data_chunks,),
            dtype="int64",
        )

        self._total_steps = 0
        self._total_episodes = 0
        print(f"[ZARR] Initialized store at '{self.zarr_path}' | obs_dim={obs_dim}, act_dim={action_dim}, img={img_shape}")

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    def append_episode(
        self,
        ep_img: Union[list, np.ndarray],
        ep_obs: Union[list, np.ndarray],
        ep_act: Union[list, np.ndarray],
    ) -> int:
        """
        Appends a completed flight episode to disk and records cumulative boundary.

        Args:
            ep_img: List or array of shape [Ep_Len, 3, H, W] uint8
            ep_obs: List or array of shape [Ep_Len, obs_dim] float32
            ep_act: List or array of shape [Ep_Len, action_dim] float32

        Returns:
            Number of steps appended.
        """
        img_arr = np.asarray(ep_img, dtype=np.uint8)
        obs_arr = np.asarray(ep_obs, dtype=np.float32)
        act_arr = np.asarray(ep_act, dtype=np.float32)

        ep_len = len(obs_arr)
        if ep_len == 0:
            return 0

        if len(img_arr) != ep_len or len(act_arr) != ep_len:
            raise ValueError(
                f"Trajectory length mismatch: img={len(img_arr)}, obs={len(obs_arr)}, act={len(act_arr)}"
            )

        # Append along temporal axis
        self.img_zarr.append(img_arr, axis=0)
        self.obs_zarr.append(obs_arr, axis=0)
        self.act_zarr.append(act_arr, axis=0)

        # Update metadata atomically for crash resilience
        self._total_steps += ep_len
        self._total_episodes += 1
        self.ep_ends_zarr.append(np.array([self._total_steps], dtype=np.int64))

        return ep_len

    def close(self):
        """Flushes indices and prints summary."""
        print(f"\n[DONE] Dataset compiled successfully at '{self.zarr_path}'")
        print(f"Total Valid Episodes: {self._total_episodes} | Total Trajectory Steps: {self._total_steps}")

    def finalize(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# PyTorch Dataset for Sliding-Window Chunk Slicing
# ---------------------------------------------------------------------------

class DroneZarrDataset(Dataset):
    """
    Sliding-window sequence dataset backed by Zarr chunked storage.
    Extracts past observation conditioning To and future action prediction horizon Ta.
    """

    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        obs_dim: Optional[int] = None,
        obs_type: str = "kin",
        episode_indices: Optional[List[int]] = None,
        stats: Optional[Dict[str, Dict[str, list]]] = None,
    ):
        self.zarr_path = zarr_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_dim = obs_dim
        self.obs_type = obs_type.lower()

        # Open Zarr archive read-only
        self.root = zarr.open(zarr_path, mode="r")
        self.data_group = self.root["data"]
        self.meta_group = self.root["meta"]

        self.img_data = self.data_group["img"]
        self.obs_data = self.data_group["obs"]
        self.action_data = self.data_group["action"]
        raw_episode_ends = self.meta_group["episode_ends"][:]

        # Construct episode boundaries [start_idx, end_idx)
        all_episodes = []
        prev_end = 0
        for ep_end in raw_episode_ends:
            all_episodes.append((prev_end, int(ep_end)))
            prev_end = int(ep_end)

        # Select episodes (for train/val partitioning)
        if episode_indices is not None:
            self.episodes = [all_episodes[i] for i in episode_indices]
        else:
            self.episodes = all_episodes

        # Precompute flat sample indices: (ep_start, ep_end, offset_t)
        self.indices = []
        for ep_start, ep_end in self.episodes:
            ep_len = ep_end - ep_start
            for t in range(ep_len):
                self.indices.append((ep_start, ep_end, t))

        # Normalization statistics
        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

    def _compute_stats(self) -> Dict[str, Dict[str, list]]:
        """Computes min, max, range normalizer bounds across training trajectories."""
        obs_arr = self.obs_data[:]
        if self.obs_dim is not None and obs_arr.shape[1] > self.obs_dim:
            obs_arr = obs_arr[:, :self.obs_dim]
        act_arr = self.action_data[:]

        obs_min = np.min(obs_arr, axis=0)
        obs_max = np.max(obs_arr, axis=0)
        obs_range = np.maximum(obs_max - obs_min, 1e-4)

        act_min = np.min(act_arr, axis=0)
        act_max = np.max(act_arr, axis=0)
        act_range = np.maximum(act_max - act_min, 1e-4)

        return {
            "obs": {
                "min": obs_min.tolist(),
                "max": obs_max.tolist(),
                "range": obs_range.tolist(),
            },
            "action": {
                "min": act_min.tolist(),
                "max": act_max.tolist(),
                "range": act_range.tolist(),
            },
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        ep_start, ep_end, t = self.indices[idx]
        global_t = ep_start + t

        # 1. Observation sequence: [global_t - obs_horizon + 1, ..., global_t]
        obs_start = max(ep_start, global_t - self.obs_horizon + 1)
        obs_end = global_t + 1

        raw_obs = self.obs_data[obs_start:obs_end]  # [Len, obs_dim]
        if self.obs_dim is not None and raw_obs.shape[1] > self.obs_dim:
            raw_obs = raw_obs[:, :self.obs_dim]

        if len(raw_obs) < self.obs_horizon:
            pad_len = self.obs_horizon - len(raw_obs)
            pad = np.repeat(raw_obs[:1], pad_len, axis=0)
            raw_obs = np.concatenate([pad, raw_obs], axis=0)

        # Normalize observation to [-1.0, 1.0]
        obs_min = np.array(self.stats["obs"]["min"], dtype=np.float32)
        obs_range = np.array(self.stats["obs"]["range"], dtype=np.float32)
        norm_obs = 2.0 * (raw_obs - obs_min) / obs_range - 1.0
        obs_tensor = torch.from_numpy(norm_obs).float()

        # 2. Action prediction sequence: [global_t, ..., global_t + pred_horizon - 1]
        act_start = global_t
        act_end = min(ep_end, global_t + self.pred_horizon)

        raw_act = self.action_data[act_start:act_end]  # [Len, action_dim]
        if len(raw_act) < self.pred_horizon:
            pad_len = self.pred_horizon - len(raw_act)
            pad = np.repeat(raw_act[-1:], pad_len, axis=0)
            raw_act = np.concatenate([raw_act, pad], axis=0)

        # Normalize action to [-1.0, 1.0]
        act_min = np.array(self.stats["action"]["min"], dtype=np.float32)
        act_range = np.array(self.stats["action"]["range"], dtype=np.float32)
        norm_act = 2.0 * (raw_act - act_min) / act_range - 1.0
        act_tensor = torch.from_numpy(norm_act).float()

        # 3. Vision / Multimodal branch
        if self.obs_type in ["rgb", "vision"]:
            raw_imgs = self.img_data[obs_start:obs_end]
            if len(raw_imgs) < self.obs_horizon:
                pad_len = self.obs_horizon - len(raw_imgs)
                pad = np.repeat(raw_imgs[:1], pad_len, axis=0)
                raw_imgs = np.concatenate([pad, raw_imgs], axis=0)
            img_tensor = torch.from_numpy(raw_imgs).float() / 255.0
            return {"obs": img_tensor, "action": act_tensor}
        elif self.obs_type == "multimodal":
            raw_imgs = self.img_data[obs_start:obs_end]
            if len(raw_imgs) < self.obs_horizon:
                pad_len = self.obs_horizon - len(raw_imgs)
                pad = np.repeat(raw_imgs[:1], pad_len, axis=0)
                raw_imgs = np.concatenate([pad, raw_imgs], axis=0)
            img_tensor = torch.from_numpy(raw_imgs).float() / 255.0
            return {
                "obs": {"rgb": img_tensor, "kin": obs_tensor},
                "action": act_tensor,
            }
        else:
            return {"obs": obs_tensor, "action": act_tensor}


# ---------------------------------------------------------------------------
# DataLoader Factory Interface
# ---------------------------------------------------------------------------

def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, Dict[str, Dict[str, list]]]:
    """
    Constructs train and validation PyTorch DataLoaders partitioned by episode.

    Args:
        cfg: Omegaconf configuration containing data, policy, and training fields.

    Returns:
        (train_loader, val_loader, dataset_stats)
    """
    zarr_path = cfg.data.zarr_path
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"Zarr dataset not found at '{zarr_path}'. Please run dataset/collect_expert.py first."
        )

    root = zarr.open(zarr_path, mode="r")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError(f"Invalid Zarr archive at '{zarr_path}': missing 'meta/episode_ends'.")

    total_episodes = len(root["meta/episode_ends"])
    if total_episodes == 0:
        raise ValueError(f"Zarr store at '{zarr_path}' contains 0 valid episodes.")

    # Partition episodes deterministically into train and val splits
    val_split = getattr(cfg.data, "val_split", 0.1)
    num_val = max(1, int(total_episodes * val_split))
    num_train = max(1, total_episodes - num_val)

    rng = np.random.RandomState(cfg.project.seed)
    shuffled_ep_indices = rng.permutation(total_episodes).tolist()
    train_ep_indices = shuffled_ep_indices[:num_train]
    val_ep_indices = shuffled_ep_indices[num_train:]

    obs_type = getattr(cfg.env, "obs_type", "kin")
    obs_horizon = cfg.policy.obs_horizon
    pred_horizon = cfg.policy.pred_horizon
    obs_dim = getattr(cfg.policy, "obs_dim", None)

    # Build train dataset and derive normalizer bounds
    train_dataset = DroneZarrDataset(
        zarr_path=zarr_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        obs_dim=obs_dim,
        obs_type=obs_type,
        episode_indices=train_ep_indices,
    )

    # Build val dataset sharing identical normalizer bounds
    val_dataset = DroneZarrDataset(
        zarr_path=zarr_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        obs_dim=obs_dim,
        obs_type=obs_type,
        episode_indices=val_ep_indices,
        stats=train_dataset.stats,
    )

    batch_size = cfg.training.batch_size
    num_workers = getattr(cfg.data, "num_workers", 0)
    pin_memory = getattr(cfg.data, "pin_memory", False) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(len(train_dataset) > batch_size),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset.stats