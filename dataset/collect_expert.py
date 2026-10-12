# dataset/collect_expert.py

import os
import shutil
import argparse
from typing import List, Optional
import numpy as np
import zarr
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf
from numcodecs import Blosc
import pybullet as p

import gymnasium as gym
import custom_drone_env
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType


# ---------------------------------------------------------------------------
# Utility: Image & Data Parsing
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
# Utility: Heading & Action Alignment
# ---------------------------------------------------------------------------

def quat_to_yaw(quat: np.ndarray) -> float:
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def compute_camera_aligned_action(
    pos: np.ndarray,
    quat: np.ndarray,
    target_pos: np.ndarray,
    computed_motor_rpms: np.ndarray,
    kp_yaw: float = 1.8,
) -> np.ndarray:
    dx, dy = target_pos[0] - pos[0], target_pos[1] - pos[1]

    current_yaw = quat_to_yaw(quat)
    target_yaw = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0.15 else current_yaw

    yaw_err = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
    yaw_cmd = float(np.clip(kp_yaw * yaw_err, -1.0, 1.0))

    thrust = float(np.clip((np.sum(computed_motor_rpms) / 4.0 - 0.5) * 2.0, -1.0, 1.0))
    roll = float(np.clip(computed_motor_rpms[0] - computed_motor_rpms[1], -1.0, 1.0))
    pitch = float(np.clip(computed_motor_rpms[2] - computed_motor_rpms[3], -1.0, 1.0))

    return np.array([thrust, roll, pitch, yaw_cmd], dtype=np.float32)


# ---------------------------------------------------------------------------
# THE EXPERT CHEAT: Reactive Obstacle Avoidance (Artificial Potential Field)
# ---------------------------------------------------------------------------

def get_obstacle_free_carrot(
    pos: np.ndarray,
    target_pos: np.ndarray,
    client_id: int,
    ray_length: float = 1.5,
    repulsion_strength: float = 2.0,
) -> np.ndarray:
    """
    Shoots invisible rays in PyBullet. If it detects an obstacle, it pushes the target
    'carrot' away, teaching the AI to maneuver around objects.
    """
    directions = np.array([
        [1.0, 0.0, 0.0],   # Forward
        [0.8, 0.5, 0.0],   # Forward-Left
        [0.8, -0.5, 0.0],  # Forward-Right
        [0.8, 0.0, 0.5],   # Forward-Up
        [0.8, 0.0, -0.5],  # Forward-Down
    ])

    to_target = target_pos - pos
    dist_to_target = np.linalg.norm(to_target)

    if dist_to_target < 0.1:
        return target_pos

    forward_dir = to_target / dist_to_target
    yaw = np.arctan2(forward_dir[1], forward_dir[0])

    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ])

    ray_starts = [pos for _ in range(len(directions))]
    ray_ends = [pos + (rot_mat @ dir_vec) * ray_length for dir_vec in directions]

    # Shoot the rays in PyBullet
    ray_results = p.rayTestBatch(ray_starts, ray_ends, physicsClientId=client_id)
    repulsive_vector = np.zeros(3)

    for i, result in enumerate(ray_results):
        object_id, link_index, hit_fraction, hit_pos, hit_normal = result
        if object_id >= 0:  # Hit an obstacle
            distance = max(hit_fraction * ray_length, 0.05)
            push_strength = repulsion_strength / (distance ** 2)
            repulsive_vector += np.array(hit_normal) * push_strength

    attractive_vector = to_target
    if np.linalg.norm(attractive_vector) > 2.0:
        attractive_vector = (attractive_vector / np.linalg.norm(attractive_vector)) * 2.0

    safe_direction = attractive_vector + repulsive_vector
    safe_direction_norm = safe_direction / (np.linalg.norm(safe_direction) + 1e-6)

    return pos + safe_direction_norm * min(0.2, dist_to_target)


# ---------------------------------------------------------------------------
# Global Waypoint Generator (The Manager)
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    def __init__(self, bounds_x=(-3.0, 3.0), bounds_y=(-3.0, 3.0), bounds_z=(0.5, 1.2)):
        self.bx, self.by, self.bz = bounds_x, bounds_y, bounds_z

    def generate_waypoints(self, num_points: int = 8) -> np.ndarray:
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
    max_steps = getattr(cfg.data, "max_episode_steps", 2000)
    action_dim = cfg.policy.action_dim
    obs_dim = cfg.policy.obs_dim  # Must be 23 for [kin(20) + relative_carrot(3)]

    # Resolve URDF path gracefully (checks config, Colab path, or None)
    cfg_urdf = getattr(cfg.env, "urdf_path", None)
    colab_urdf = "/content/drive/MyDrive/DroneDiffusion/test_world.urdf"
    if cfg_urdf and os.path.exists(cfg_urdf):
        urdf_path = cfg_urdf
    elif os.path.exists(colab_urdf):
        urdf_path = colab_urdf
    else:
        urdf_path = cfg_urdf or ""

    print(f"[COLLECT] Initializing Goal-Conditioned FPV collection pipeline...")
    if urdf_path and os.path.exists(urdf_path):
        print(f"[COLLECT] World URDF resolved: '{urdf_path}'")
    else:
        print("[COLLECT] Running without external URDF world.")

    if os.path.exists(zarr_path):
        confirm = input(f"Zarr store '{zarr_path}' already exists. Overwrite? (y/n): ")
        if confirm.lower() == "y":
            shutil.rmtree(zarr_path)
        else:
            print("Aborting.")
            return

    # 1. Setup Progressive Zarr Storage
    root = zarr.open(zarr_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    img_compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)
    vec_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    img_zarr = data_group.create_dataset(
        "img",
        shape=(0, 3, 224, 224),
        chunks=(500, 3, 224, 224),
        dtype="uint8",
        compressor=img_compressor,
    )
    obs_zarr = data_group.create_dataset(
        "obs",
        shape=(0, obs_dim),
        chunks=(1000, obs_dim),
        dtype="float32",
        compressor=vec_compressor,
    )
    act_zarr = data_group.create_dataset(
        "action",
        shape=(0, action_dim),
        chunks=(1000, action_dim),
        dtype="float32",
        compressor=vec_compressor,
    )
    ep_ends_zarr = meta_group.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(1000,),
        dtype="int64",
    )

    # 2. Initialize Environment
    env_kwargs = {
        "id": "CustomDrone-v0",
        "gui": getattr(cfg.env, "gui", False),
        "obs": ObservationType.KIN,
        "pyb_freq": getattr(cfg.env, "physics_hz", 240),
        "ctrl_freq": getattr(cfg.env, "physics_hz", 240),
        "enable_ipc": False,
        "initial_xyzs": np.array([[0.0, 0.0, 1.0]]),
    }
    if urdf_path:
        env_kwargs["urdf_path"] = urdf_path

    env = gym.make(**env_kwargs)

    env.unwrapped._computeInfo = lambda: {}
    env.unwrapped._computeReward = lambda: 0.0
    env.unwrapped._computeTerminated = lambda: False
    env.unwrapped._computeTruncated = lambda: False
    env.unwrapped.IMG_RES = np.array([224, 224])

    pb_client = env.unwrapped.CLIENT
    pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)

    cumulative_steps = 0
    dt = 1.0 / float(getattr(cfg.env, "physics_hz", 240))

    # 3. Episode Rollout Loop
    for ep in tqdm(range(num_episodes), desc="Collecting Expert Flights"):

        # --- 1. RANDOM SPAWN ---
        start_pos = np.array([
            np.random.uniform(-2.5, 2.5),
            np.random.uniform(-2.5, 2.5),
            np.random.uniform(0.5, 1.2),
        ])
        env.unwrapped.INIT_XYZS = np.array([start_pos])
        obs_data, info = env.reset()

        # --- 2. RANDOM 3D SPHERE TARGET (3-Meter Radius) ---
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1.0, 1.0))
        r = 3.0 * np.cbrt(np.random.uniform(0, 1.0))

        target_offset = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        global_target = start_pos + target_offset
        global_target[2] = np.clip(global_target[2], 0.5, 1.5)

        ep_obs, ep_act, ep_img = [], [], []

        for step in range(max_steps):
            current_kin = env.unwrapped._getDroneStateVector(0)

            pos = current_kin[0:3]
            quat = current_kin[3:7]
            vel = current_kin[10:13]
            ang_vel = current_kin[13:16]

            # --- 3. DYNAMIC EPISODE END ---
            if np.linalg.norm(global_target - pos) < 0.3:
                break

            safe_carrot = get_obstacle_free_carrot(pos, global_target, pb_client)
            relative_carrot = safe_carrot - pos

            dx, dy = relative_carrot[0], relative_carrot[1]
            desired_yaw = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0.15 else quat_to_yaw(quat)

            computed_rpms, _, _ = pid_controller.computeControlFromState(
                control_timestep=dt,
                state=np.hstack([pos, quat, vel, ang_vel, np.zeros(4)]),
                target_pos=safe_carrot,
                target_rpy=np.array([0.0, 0.0, desired_yaw]),
                target_vel=np.zeros(3),
            )

            # --- THE EXPLOSION FIX: Normalize RPMs to [-1.0, 1.0] ---
            max_rpm = env.unwrapped.MAX_RPM
            normalized_rpms = (computed_rpms / max_rpm) * 2.0 - 1.0

            action = compute_camera_aligned_action(
                pos=pos,
                quat=quat,
                target_pos=safe_carrot,
                computed_motor_rpms=normalized_rpms,
            )

            # --- RENDER AND SAVE DATA AT 20 FPS (every 12 physics steps) ---
            if step % 12 == 0:
                rgb, dep, seg = env.unwrapped._getDroneImages(0)
                kinematics_with_goal = np.hstack([current_kin[:20], relative_carrot])

                ep_obs.append(kinematics_with_goal.astype(np.float32))
                ep_img.append(process_fpv_image(rgb))
                ep_act.append(action)

            # Step PyBullet simulation with normalized motor actions
            sim_action = np.array([normalized_rpms], dtype=np.float32)
            obs_data, reward, terminated, truncated, info = env.step(sim_action)

            if terminated or truncated:
                break

        # 4. Progressively Write to Disk
        if len(ep_obs) > 50:
            img_zarr.append(np.array(ep_img, dtype=np.uint8))
            obs_zarr.append(np.array(ep_obs, dtype=np.float32))
            act_zarr.append(np.array(ep_act, dtype=np.float32))

            cumulative_steps += len(ep_obs)
            ep_ends_zarr.append(np.array([cumulative_steps], dtype=np.int64))

    env.close()

    print(f"\n[DONE] Dataset compiled successfully at '{zarr_path}'.")
    print(f"Total Trajectory Steps: {cumulative_steps}")
    print(f"Total Valid Episodes:   {len(ep_ends_zarr)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    collect_expert_trajectories(cfg)