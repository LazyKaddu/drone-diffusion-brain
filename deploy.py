# deploy.py

import os
import time
import argparse
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
import gymnasium as gym

# Registers "CustomDrone-v0" with Gymnasium
import custom_drone_env
from models.policy import DroneDiffusionPolicy


class RealTimeNormalizer:
    """In-memory normalizer using statistics serialized inside the checkpoint."""

    def __init__(self, stats: Dict[str, Dict[str, np.ndarray]], device: torch.device):
        self.obs_min = torch.tensor(stats["obs"]["min"], dtype=torch.float32, device=device)
        self.obs_range = torch.tensor(stats["obs"]["range"], dtype=torch.float32, device=device)
        self.act_min = stats["action"]["min"]
        self.act_range = stats["action"]["range"]

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Linear mapping to [-1.0, 1.0]."""
        return 2.0 * (obs - self.obs_min) / self.obs_range - 1.0

    def unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Inverse mapping from [-1.0, 1.0] back to physical action limits."""
        return (action_norm + 1.0) / 2.0 * self.act_range + self.act_min


def load_deployment_policy(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[DroneDiffusionPolicy, RealTimeNormalizer, dict]:
    """Loads model weights (preferring the EMA shadow) and normalization limits."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[DEPLOY] Loading model checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct configuration from serialized dictionary
    cfg = OmegaConf.create(ckpt["config"])

    # Instantiate policy
    policy = DroneDiffusionPolicy(cfg).to(device)

    # Load EMA weights if present, falling back to base model weights
    if ckpt.get("ema_state_dict") is not None:
        print("[DEPLOY] Using smoothed Exponential Moving Average (EMA) weights.")
        policy.load_state_dict(ckpt["ema_state_dict"])
    else:
        print("[DEPLOY] Using raw training weights (no EMA found).")
        policy.load_state_dict(ckpt["model_state_dict"])

    policy.eval()

    # Build normalizer
    normalizer = RealTimeNormalizer(ckpt["stats"], device=device)

    return policy, normalizer, cfg


def run_deployment_loop(
    checkpoint_path: str,
    gui: bool = True,
    max_steps: int = 2000,
    device_name: str = "cuda",
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    policy, normalizer, cfg = load_deployment_policy(checkpoint_path, device)

    pred_horizon = cfg.policy.pred_horizon   # Ta = 16
    exec_horizon = cfg.policy.exec_horizon   # Te = 4
    obs_horizon = cfg.policy.obs_horizon     # To = 1
    control_hz = cfg.env.control_hz          # 20 Hz (50ms interval)
    dt_target = 1.0 / control_hz

    # 1. Initialize Gymnasium Environment
    env = gym.make(
        cfg.env.id,
        gui=gui,
        obs=cfg.env.obs_type,
        act="pid",
        enable_ipc=True,
    )

    raw_obs, info = env.reset()
    current_kin = raw_obs["0"] if isinstance(raw_obs, dict) else raw_obs

    # Observation FIFO buffer to maintain [To, obs_dim] history
    obs_queue = deque(maxlen=obs_horizon)
    for _ in range(obs_horizon):
        obs_queue.append(current_kin)

    # 2. GPU Warm-up (Compiles CUDA kernels and prevents latency spikes on step 0)
    print("[DEPLOY] Running GPU warm-up forward pass...")
    with torch.no_grad():
        dummy_obs = torch.zeros((1, obs_horizon, cfg.policy.obs_dim), device=device)
        _ = policy.predict_action(dummy_obs)

    print(f"[DEPLOY] Starting closed-loop execution at {control_hz} Hz...")
    print(f"[CONFIG] Prediction Horizon (Ta): {pred_horizon} | Execution Horizon (Te): {exec_horizon}")

    step_count = 0
    latency_records = []

    try:
        while step_count < max_steps:
            cycle_start = time.perf_counter()

            # -------------------------------------------------------------
            # Step A: Construct Observation Tensor
            # -------------------------------------------------------------
            obs_tensor = torch.tensor(
                np.array(obs_queue), dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1, To, obs_dim]

            # Normalize to [-1.0, 1.0]
            norm_obs = normalizer.normalize_obs(obs_tensor)

            # -------------------------------------------------------------
            # Step B: Fast Diffusion Denoising (10-Step DDIM)
            # -------------------------------------------------------------
            t_infer_start = time.perf_counter()
            with torch.no_grad():
                # Denoise entire chunk [1, 16, 4]
                action_chunk_norm = policy.predict_action(norm_obs)
            inference_ms = (time.perf_counter() - t_infer_start) * 1000.0
            latency_records.append(inference_ms)

            # Extract actions to host: [16, 4]
            actions_norm = action_chunk_norm.squeeze(0).cpu().numpy()

            # Inverse-normalize to physical motor ranges
            actions_phys = normalizer.unnormalize_action(actions_norm)

            # -------------------------------------------------------------
            # Step C: Receding Horizon Execution (First Te = 4 steps)
            # -------------------------------------------------------------
            for i in range(exec_horizon):
                step_start = time.perf_counter()

                # Dispatch single action slice to PyBullet
                act_command = actions_phys[i]
                action_payload = {"0": act_command} if isinstance(raw_obs, dict) else act_command

                raw_obs, reward, terminated, truncated, info = env.step(action_payload)
                step_count += 1

                # Update state queue
                current_kin = raw_obs["0"] if isinstance(raw_obs, dict) else raw_obs
                obs_queue.append(current_kin)

                # Reset on crash or boundary breach
                if terminated or truncated:
                    print(f"[RESET] Episode boundary at global step {step_count}. Resetting world...")
                    raw_obs, info = env.reset()
                    current_kin = raw_obs["0"] if isinstance(raw_obs, dict) else raw_obs
                    for _ in range(obs_horizon):
                        obs_queue.append(current_kin)
                    break

                # Enforce consistent control cadence
                elapsed = time.perf_counter() - step_start
                sleep_time = dt_target - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # -------------------------------------------------------------
            # Step D: Cycle Diagnostics & Profiling
            # -------------------------------------------------------------
            cycle_duration = (time.perf_counter() - cycle_start) * 1000.0
            if step_count % (exec_horizon * 5) == 0:
                print(
                    f"[LOOP] Step: {step_count:04d} | "
                    f"DDIM Inference: {inference_ms:.2f} ms | "
                    f"Te Cycle Time: {cycle_duration:.1f} ms | "
                    f"Budget (<50ms): {'OK' if inference_ms < 50.0 else 'EXCEEDED'}"
                )

    except KeyboardInterrupt:
        print("\n[STOP] Deployment interrupted by user.")

    finally:
        env.close()
        if latency_records:
            latencies = np.array(latency_records)
            print("\n--- Latency Benchmark Summary ---")
            print(f"Mean DDIM Inference:  {np.mean(latencies):.2f} ms")
            print(f"P95 DDIM Inference:   {np.percentile(latencies, 95):.2f} ms")
            print(f"P99 DDIM Inference:   {np.percentile(latencies, 99):.2f} ms")
            print(f"Min / Max:            {np.min(latencies):.2f} ms / {np.max(latencies):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Drone Diffusion Policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_ema.pt",
        help="Path to trained policy checkpoint (.pt)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Render PyBullet simulation GUI",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute target: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3000,
        help="Total environment control steps before termination",
    )

    args = parser.parse_args()
    run_deployment_loop(
        checkpoint_path=args.checkpoint,
        gui=args.gui,
        max_steps=args.max_steps,
        device_name=args.device,
    )