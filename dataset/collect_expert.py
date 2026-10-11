# dataset/collect_expert.py

import os
import argparse
from typing import List
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import cv2  

from dataset.zarr_dataset import ZarrDataWriter

import gymnasium as gym
import custom_drone_env
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


# ---------------------------------------------------------------------------
# Image & Data Parsing Utilities
# ---------------------------------------------------------------------------

def process_fpv_image(raw_img: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Converts PyBullet [H, W, 4] RGBA to PyTorch [3, H, W] RGB uint8."""
    rgb = raw_img[..., :3]
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    transposed = np.transpose(resized, (2, 0, 1))
    return transposed.astype(np.uint8)


def extract_drone_obs(obs):
    """Safely extracts the specific drone's observation dictionary."""
    if isinstance(obs, dict) and "0" in obs:
        return obs["0"]
    return obs


# ---------------------------------------------------------------------------
# Heading & Control Mathematics
# ---------------------------------------------------------------------------

def quat_to_yaw(quat: np.ndarray) -> float:
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def compute_camera_aligned_action(
    pos: np.ndarray, quat: np.ndarray, target_pos: np.ndarray,
    computed_motor_rpms: np.ndarray, kp_yaw: float = 1.8,
) -> np.ndarray:
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    
    current_yaw = quat_to_yaw(quat)
    target_yaw = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0.15 else current_yaw
    
    yaw_err = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
    yaw_cmd = float(np.clip(kp_yaw * yaw_err, -1.0, 1.0))

    thrust = float(np.clip((np.sum(computed_motor_rpms) / 4.0 - 0.5) * 2.0, -1.0, 1.0))
    roll = float(np.clip(computed_motor_rpms[0] - computed_motor_rpms[1], -1.0, 1.0))
    pitch = float(np.clip(computed_motor_rpms[2] - computed_motor_rpms[3], -1.0, 1.0))

    return np.array([thrust, roll, pitch, yaw_cmd], dtype=np.float32)


# ---------------------------------------------------------------------------
# Procedural Trajectory Generation
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    def __init__(self, bounds_x=(-2.0, 2.0), bounds_y=(-2.0, 2.0), bounds_z=(0.5, 2.0)):
        self.bx, self.by, self.bz = bounds_x, bounds_y, bounds_z

    def generate_waypoints(self, num_points: int = 6) -> np.ndarray:
        pts = np.zeros((num_points, 3))
        pts[0] = np.array([0.0, 0.0, 1.0])
        for i in range(1, num_points):
            pts[i] = [
                np.random.uniform(*self.bx),
                np.random.uniform(*self.by),
                np.random.uniform(*self.bz),
            ]
        return pts


# ---------------------------------------------------------------------------
# Main Data Collection Pipeline
# ---------------------------------------------------------------------------

def collect_expert_trajectories(cfg):
    zarr_path = cfg.data.zarr_path
    num_episodes = cfg.data.num_episodes
    max_steps = cfg.data.max_episode_steps
    action_dim = cfg.policy.action_dim
    obs_dim = cfg.policy.obs_dim

    print(f"[COLLECT] Initializing camera-aligned data collection pipeline...")
    
    if os.path.exists(zarr_path):
        confirm = input(f"Zarr store '{zarr_path}' already exists. Overwrite? (y/n): ")
        if confirm.lower() != "y":
            print("Aborting.")
            return

    # 1. Setup Zarr Storage Writer
    writer = ZarrDataWriter(
        zarr_path=zarr_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        img_shape=(3, 224, 224),
        overwrite=True,
    )

    # 2. Initialize Environment
    env = gym.make(
        cfg.env.id,
        gui=False,
        obs="rgb",        # <-- FIX: Hardcode to 'rgb' to guarantee camera rendering
        act="pid",
        enable_ipc=False,
    )

    pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    traj_gen = TrajectoryGenerator()

    cumulative_steps = 0
    dt = 1.0 / cfg.env.physics_hz

    # 3. Episode Rollout Loop
    for ep in tqdm(range(num_episodes), desc="Collecting Expert Flights"):
        obs_data, info = env.reset()

        waypoints = traj_gen.generate_waypoints(num_points=8)
        current_wp_idx = 1
        target_pos = waypoints[current_wp_idx]

        ep_obs, ep_act, ep_img = [], [], []

        for step in range(max_steps):
            # FIX: Safely parse observation dict
            drone_obs = extract_drone_obs(obs_data)
            raw_image = drone_obs["rgb"]
            current_kin = drone_obs["kin"]
            
            pos = current_kin[0:3]
            quat = current_kin[3:7]
            vel = current_kin[10:13]
            ang_vel = current_kin[13:16]
            
            processed_img = process_fpv_image(raw_image)

            # Advance waypoint if drone is within 25 cm of target
            dx, dy = target_pos[0] - pos[0], target_pos[1] - pos[1]
            if np.linalg.norm(target_pos - pos) < 0.25 and current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                target_pos = waypoints[current_wp_idx]
                dx, dy = target_pos[0] - pos[0], target_pos[1] - pos[1]

            desired_yaw = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0.15 else quat_to_yaw(quat)

            # Compute analytic expert control using PID
            computed_rpms, _, _ = pid_controller.computeControlFromState(
                control_timestep=dt,
                state=np.hstack([pos, quat, vel, ang_vel, np.zeros(4)]),
                target_pos=target_pos,
                target_rpy=np.array([0.0, 0.0, desired_yaw]), 
                target_vel=np.zeros(3),
            )

            action = compute_camera_aligned_action(
                pos=pos, quat=quat, target_pos=target_pos,
                computed_motor_rpms=computed_rpms,
            )

            # Include target goal coordinate vector if obs_dim is 23
            if obs_dim == 23 and len(current_kin) >= 20:
                full_kin = np.hstack([current_kin[:20], target_pos]).astype(np.float32)
            else:
                full_kin = current_kin[:obs_dim].astype(np.float32)

            ep_obs.append(full_kin)
            ep_img.append(processed_img)
            ep_act.append(action)

            # Step PyBullet simulation
            action_payload = {"0": action} if isinstance(obs_data, dict) and "0" in obs_data else action
            obs_data, reward, terminated, truncated, info = env.step(action_payload)

            if terminated or truncated:
                break

        # 4. Progressively Write to Disk (Prevents RAM Crash)
        if len(ep_obs) > 50:
            writer.append_episode(ep_img=ep_img, ep_obs=ep_obs, ep_act=ep_act)

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    collect_expert_trajectories(cfg)