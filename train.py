# train.py

import os
import math
import copy
import argparse
from typing import Dict, Tuple
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dataset.zarr_dataset import get_dataloaders
from models.policy import DroneDiffusionPolicy


class EMAModel:
    """
    Exponential Moving Average (EMA) shadow model.
    Maintains a smoothed copy of model parameters to stabilize diffusion sampling.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.averaged_model = copy.deepcopy(model).eval()
        for param in self.averaged_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module):
        # Update weights: shadow = decay * shadow + (1 - decay) * new
        for ema_param, model_param in zip(self.averaged_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

        # Copy non-trainable buffers directly (e.g., normalizer statistics, running state)
        for ema_buffer, model_buffer in zip(self.averaged_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)

    def state_dict(self) -> dict:
        return self.averaged_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.averaged_model.load_state_dict(state_dict)


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Constructs a learning rate scheduler with linear warmup followed by cosine annealing decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(policy: nn.Module, val_loader, device: torch.device) -> float:
    """
    Runs full validation pass and computes mean diffusion MSE loss.
    """
    policy.eval()
    total_val_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        # Move tensors to device
        batch_dev = {
            "obs": batch["obs"].to(device, non_blocking=True),
            "action": batch["action"].to(device, non_blocking=True),
        }
        loss = policy.compute_loss(batch_dev)
        total_val_loss += loss.item()
        num_batches += 1

    return total_val_loss / max(1, num_batches)


def train(cfg):
    # Set deterministic seeds
    torch.manual_seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.project.seed)

    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    print(f"[INIT] Active compute device: {device}")
    print(f"[INIT] Loading trajectory dataset from: {cfg.data.zarr_path}")

    # 1. Dataset & Loaders
    train_loader, val_loader, dataset_stats = get_dataloaders(cfg)
    print(f"[DATA] Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 2. Instantiate Policy Model
    policy = DroneDiffusionPolicy(cfg).to(device)

    # 3. Instantiate EMA Shadow Model
    ema: EMAModel = None
    if cfg.training.use_ema:
        ema = EMAModel(policy, decay=cfg.training.ema_decay)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        policy.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_training_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_steps=cfg.training.lr_warmup_steps,
        total_steps=total_training_steps,
    )

    # Optional Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if cfg.training.resume_checkpoint:
        print(f"[RESUME] Loading checkpoint: {cfg.training.resume_checkpoint}")
        ckpt = torch.load(cfg.training.resume_checkpoint, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # 5. Training Loop
    print(f"[START] Commencing training across {cfg.training.num_epochs} epochs...")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        policy.train()
        epoch_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:03d}/{cfg.training.num_epochs:03d}")
        for batch in pbar:
            batch_dev = {
                "obs": batch["obs"].to(device, non_blocking=True),
                "action": batch["action"].to(device, non_blocking=True),
            }

            # Forward pass: sample random noise, add to action chunk, predict with U-Net
            loss = policy.compute_loss(batch_dev)

            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.training.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip_norm)

            optimizer.step()
            scheduler.step()

            # Update EMA weights
            if ema is not None:
                ema.step(policy)

            # Metrics
            global_step += 1
            epoch_train_loss += loss.item()
            curr_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{curr_lr:.2e}",
            })

        avg_train_loss = epoch_train_loss / len(train_loader)

        # 6. Evaluation Step (Evaluating the EMA model shadow)
        eval_target = ema.averaged_model if ema is not None else policy
        val_loss = evaluate(eval_target, val_loader, device)

        print(
            f"[METRIC] Epoch {epoch + 1:03d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss (EMA): {val_loss:.5f} | "
            f"LR: {curr_lr:.2e}"
        )

        # 7. Checkpointing
        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": policy.state_dict(),
            "ema_state_dict": ema.state_dict() if ema else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "stats": dataset_stats,  # Bundled normalizer stats for standalone deployment
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Save latest checkpoint
        latest_path = os.path.join(cfg.training.checkpoint_dir, "latest.pt")
        torch.save(checkpoint_data, latest_path)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data["best_val_loss"] = best_val_loss
            best_path = os.path.join(cfg.training.checkpoint_dir, "best_ema.pt")
            torch.save(checkpoint_data, best_path)
            print(f"[CHECKPOINT] Best validation loss reached ({val_loss:.5f}). Saved to {best_path}")

        # Periodic checkpoint snapshots
        if (epoch + 1) % cfg.training.save_interval == 0:
            interval_path = os.path.join(
                cfg.training.checkpoint_dir, f"policy_epoch_{epoch + 1:03d}.pt"
            )
            torch.save(checkpoint_data, interval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone Diffusion Policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to Hydra YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    train(cfg)