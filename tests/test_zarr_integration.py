# test_zarr_integration.py

import os
import shutil
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset.zarr_dataset import ZarrDataWriter, DroneZarrDataset, get_dataloaders
from models.policy import DroneDiffusionPolicy


def test_zarr_writer_and_reader():
    test_dir = "data/test_trajectories.zarr"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("--- 1. Testing ZarrDataWriter (23-dim) ---")
    obs_dim = 23
    action_dim = 4
    img_shape = (3, 224, 224)

    writer = ZarrDataWriter(
        zarr_path=test_dir,
        obs_dim=obs_dim,
        action_dim=action_dim,
        img_shape=img_shape,
        overwrite=True,
    )

    # Generate 3 dummy episodes with different lengths
    ep_lengths = [60, 75, 55]
    for i, ep_len in enumerate(ep_lengths):
        ep_img = np.random.randint(0, 255, size=(ep_len, *img_shape), dtype=np.uint8)
        ep_obs = np.random.randn(ep_len, obs_dim).astype(np.float32)
        ep_act = np.random.uniform(-1.0, 1.0, size=(ep_len, action_dim)).astype(np.float32)
        writer.append_episode(ep_img, ep_obs, ep_act)

    writer.close()

    assert writer.total_episodes == 3, f"Expected 3 episodes, got {writer.total_episodes}"
    expected_total_steps = sum(ep_lengths)
    assert writer.total_steps == expected_total_steps, f"Expected {expected_total_steps} steps, got {writer.total_steps}"
    print(f"PASS: Writer stored {writer.total_episodes} episodes ({writer.total_steps} frames) successfully.")

    print("\n--- 2. Testing DroneZarrDataset (23-dim) ---")
    dataset = DroneZarrDataset(
        zarr_path=test_dir,
        obs_horizon=1,
        pred_horizon=16,
        obs_dim=obs_dim,
        obs_type="kin",
    )

    assert len(dataset) == expected_total_steps, f"Dataset length {len(dataset)} != expected {expected_total_steps}"
    sample = dataset[0]
    assert "obs" in sample and "action" in sample
    assert sample["obs"].shape == (1, obs_dim), f"Obs shape: {sample['obs'].shape}"
    assert sample["action"].shape == (16, action_dim), f"Action shape: {sample['action'].shape}"
    print(f"PASS: DroneZarrDataset sample 0 has obs shape {sample['obs'].shape} and action shape {sample['action'].shape}")

    print("\n--- 3. Testing get_dataloaders factory ---")
    cfg = OmegaConf.load("configs/default.yaml")
    # Temporarily point config to test dataset
    cfg.data.zarr_path = test_dir
    cfg.training.batch_size = 16

    train_loader, val_loader, stats = get_dataloaders(cfg)
    assert "obs" in stats and "action" in stats
    print(f"PASS: get_dataloaders returned {len(train_loader)} train batches, {len(val_loader)} val batches.")

    batch = next(iter(train_loader))
    assert batch["obs"].shape == (16, 1, obs_dim), f"Batch obs shape: {batch['obs'].shape}"
    assert batch["action"].shape == (16, 16, action_dim), f"Batch action shape: {batch['action'].shape}"
    print(f"PASS: Dataloader yielded batch obs={batch['obs'].shape}, action={batch['action'].shape}")

    print("\n--- 4. Testing DroneDiffusionPolicy compute_loss & predict_action ---")
    device = torch.device("cpu")
    cfg.project.device = "cpu"
    policy = DroneDiffusionPolicy(cfg).to(device)

    # Compute training loss
    batch_dev = {
        "obs": batch["obs"].to(device),
        "action": batch["action"].to(device),
    }
    loss = policy.compute_loss(batch_dev)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"PASS: policy.compute_loss succeeded with loss={loss.item():.4f}")

    # Predict actions
    with torch.no_grad():
        pred_action = policy.predict_action(batch_dev["obs"][:2], num_inference_steps=2)
        assert pred_action.shape == (2, 16, 4), f"Predicted action shape: {pred_action.shape}"
        assert torch.all(pred_action >= -1.0) and torch.all(pred_action <= 1.0)
    print(f"PASS: policy.predict_action succeeded with output shape {pred_action.shape}")

    # Clean up test zarr
    shutil.rmtree(test_dir)

    print("\n--- 5. Testing Real Dataset data/expert_trajectories.zarr ---")
    real_cfg = OmegaConf.load("configs/default.yaml")
    real_cfg.data.zarr_path = "data/expert_trajectories.zarr"
    if os.path.exists(real_cfg.data.zarr_path):
        real_tl, real_vl, real_stats = get_dataloaders(real_cfg)
        real_batch = next(iter(real_tl))
        real_loss = policy.compute_loss({
            "obs": real_batch["obs"].to(device),
            "action": real_batch["action"].to(device),
        })
        print(f"PASS: Real dataset batch ({real_batch['obs'].shape}) loss computed: {real_loss.item():.4f}")
    else:
        print("Note: data/expert_trajectories.zarr not found, skipping real dataset test.")

    print("\nALL INTEGRATION AND MODEL ALIGNMENT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_zarr_writer_and_reader()
