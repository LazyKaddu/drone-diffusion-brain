# tests/test_dataset.py

import os
import pytest
import zarr
import numpy as np
import torch

from dataset.zarr_dataset import ZarrDataWriter, DroneZarrDataset, get_dataloaders


class TestZarrDataWriter:
    def test_writer_lifecycle(self, tmp_path):
        store_path = str(tmp_path / "lifecycle.zarr")
        writer = ZarrDataWriter(
            zarr_path=store_path,
            obs_dim=23,
            action_dim=4,
            img_shape=(3, 224, 224),
            img_chunks=20,
            data_chunks=50,
            overwrite=True,
        )

        ep_img = np.zeros((15, 3, 224, 224), dtype=np.uint8)
        ep_obs = np.ones((15, 23), dtype=np.float32)
        ep_act = np.full((15, 4), 0.5, dtype=np.float32)

        steps = writer.append_episode(ep_img, ep_obs, ep_act)
        assert steps == 15
        assert writer.total_steps == 15
        assert writer.total_episodes == 1

        writer.close()

        # Reopen and inspect raw Zarr v2 array properties
        root = zarr.open(store_path, mode="r")
        assert root["data/img"].shape == (15, 3, 224, 224)
        assert root["data/obs"].shape == (15, 23)
        assert root["data/action"].shape == (15, 4)
        assert root["meta/episode_ends"][:].tolist() == [15]

    def test_writer_overwrite_handling(self, tmp_path):
        store_path = str(tmp_path / "overwrite.zarr")
        writer1 = ZarrDataWriter(zarr_path=store_path, overwrite=True)
        writer1.close()

        # Attempting to initialize without overwrite flag must raise FileExistsError
        with pytest.raises(FileExistsError):
            ZarrDataWriter(zarr_path=store_path, overwrite=False)

        # Initializing with overwrite=True succeeds
        writer2 = ZarrDataWriter(zarr_path=store_path, overwrite=True)
        writer2.close()

    def test_writer_empty_and_mismatched_inputs(self, tmp_path):
        store_path = str(tmp_path / "edge_cases.zarr")
        writer = ZarrDataWriter(zarr_path=store_path, overwrite=True)

        # Empty episode should return 0 and not modify state
        assert writer.append_episode([], [], []) == 0
        assert writer.total_steps == 0

        # Mismatched lengths should raise ValueError
        with pytest.raises(ValueError):
            writer.append_episode(
                np.zeros((10, 3, 224, 224), dtype=np.uint8),
                np.zeros((8, 23), dtype=np.float32),
                np.zeros((10, 4), dtype=np.float32),
            )
        writer.close()

    def test_writer_context_manager(self, tmp_path):
        store_path = str(tmp_path / "ctx.zarr")
        with ZarrDataWriter(zarr_path=store_path, overwrite=True) as writer:
            writer.append_episode(
                np.zeros((5, 3, 224, 224), dtype=np.uint8),
                np.zeros((5, 23), dtype=np.float32),
                np.zeros((5, 4), dtype=np.float32),
            )
            assert writer.total_steps == 5

        root = zarr.open(store_path, mode="r")
        assert root["meta/episode_ends"][:].tolist() == [5]


class TestDroneZarrDataset:
    def test_sliding_window_and_shapes(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=1,
            pred_horizon=16,
            obs_type="kin",
        )

        assert len(dataset) == temp_zarr_store["total_steps"]
        sample = dataset[0]
        assert "obs" in sample and "action" in sample
        assert sample["obs"].shape == (1, 23)
        assert sample["action"].shape == (16, 4)
        assert sample["obs"].dtype == torch.float32
        assert sample["action"].dtype == torch.float32

    def test_boundary_padding_no_cross_episode_leakage(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        ep0_len = temp_zarr_store["ep_lengths"][0]

        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=2,
            pred_horizon=16,
            obs_type="kin",
        )

        # Index 0: timestep 0 of episode 0. Obs horizon=2 requires padding first step
        sample_start = dataset[0]
        assert sample_start["obs"].shape == (2, 23)
        # Verify first observation is repeated/padded
        assert torch.allclose(sample_start["obs"][0], sample_start["obs"][1])

        # Last timestep of episode 0: index ep0_len - 1
        sample_end = dataset[ep0_len - 1]
        assert sample_end["action"].shape == (16, 4)
        # In diffusion policy padding, the tail steps are padded with the last physical action
        last_action = sample_end["action"][-1]
        penultimate_action = sample_end["action"][-2]
        assert torch.allclose(last_action, penultimate_action)

    def test_vision_observation_mode(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=1,
            pred_horizon=16,
            obs_type="rgb",
        )

        sample = dataset[0]
        assert sample["obs"].shape == (1, 3, 224, 224)
        assert sample["obs"].dtype == torch.float32
        # Image pixel normalization: [0.0, 1.0]
        assert sample["obs"].min() >= 0.0
        assert sample["obs"].max() <= 1.0

    def test_multimodal_observation_mode(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=1,
            pred_horizon=16,
            obs_type="multimodal",
        )

        sample = dataset[0]
        assert isinstance(sample["obs"], dict)
        assert "rgb" in sample["obs"] and "kin" in sample["obs"]
        assert sample["obs"]["rgb"].shape == (1, 3, 224, 224)
        assert sample["obs"]["kin"].shape == (1, 23)
        assert sample["action"].shape == (16, 4)

    def test_defensive_obs_dim_slicing(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        # Explicitly request 20 dimensions from 23-dim storage
        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=1,
            pred_horizon=16,
            obs_dim=20,
            obs_type="kin",
        )

        sample = dataset[0]
        assert sample["obs"].shape == (1, 20)

    def test_normalizer_statistics(self, temp_zarr_store):
        store_path = temp_zarr_store["path"]
        dataset = DroneZarrDataset(
            zarr_path=store_path,
            obs_horizon=1,
            pred_horizon=16,
            obs_type="kin",
        )

        stats = dataset.stats
        assert "obs" in stats and "action" in stats
        for key in ["min", "max", "range"]:
            assert key in stats["obs"]
            assert key in stats["action"]
            assert len(stats["obs"][key]) == 23
            assert len(stats["action"][key]) == 4

        # Verify normalized actions are bounded within [-1.0, 1.0]
        for i in range(min(5, len(dataset))):
            act = dataset[i]["action"]
            assert act.min() >= -1.01 and act.max() <= 1.01


class TestGetDataLoaders:
    def test_dataloader_partition_and_batches(self, mock_cfg, temp_zarr_store):
        mock_cfg.data.zarr_path = temp_zarr_store["path"]
        mock_cfg.training.batch_size = 8

        train_loader, val_loader, stats = get_dataloaders(mock_cfg)

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert "obs" in stats and "action" in stats

        # Inspect a train batch
        train_batch = next(iter(train_loader))
        assert train_batch["obs"].shape == (8, 1, 23)
        assert train_batch["action"].shape == (8, 16, 4)

        # Inspect a val batch
        val_batch = next(iter(val_loader))
        assert val_batch["obs"].shape[1:] == (1, 23)
        assert val_batch["action"].shape[1:] == (16, 4)
