# dataset/collect_expert.py

import os
import shutil
import argparse
from typing import List
import numpy as np
import zarr
from tqdm import tqdm
from omegaconf import OmegaConf

import gymnasium as gym
import custom_drone_env  # Registers "CustomDrone-v0"
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


# ---------------------------------------------------------------------------
# Heading & Control Mathematics
# ---------------------------------------------------------------------------

def quat_to_yaw(quat: np.ndarray) -> float:
    """Extracts scalar yaw (heading) angle in radians from [x, y, z, w] quaternion."""
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
    """
    Computes a 4-channel action [Thrust, Roll, Pitch, Yaw] keeping the
    drone's nose and camera aligned with the target waypoint.
    """
    # 1. Target vector in horizontal plane
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    dist_horizontal = np.hypot(dx, dy)

    # 2. Extract current heading and desired heading
    current_yaw = quat_to_yaw(quat)
    
    # If hovering directly above/below waypoint, maintain current heading
    if dist_horizontal > 0.15:
        target_yaw = np.arctan2(dy, dx)
    else:
        target_yaw = current_yaw

    # 3. Shortest-path angular error wrapped to [-pi, pi]
    yaw_err = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))

    # 4. Proportional yaw rate command
    yaw_cmd = float(np.clip(kp_yaw * yaw_err, -1.0, 1.0))

    # 5. Motor RPM to normalized thrust/attitude commands [-1.0, 1.0]
    thrust = float(np.clip((np.sum(computed_motor_rpms) / 4.0 - 0.5) * 2.0, -1.0, 1.0))
    roll = float(np.clip(computed_motor_rpms[0] - computed_motor_rpms[1], -1.0, 1.0))
    pitch = float(np.clip(computed_motor_rpms[2] - computed_motor_rpms[3], -1.0, 1.0))

    return np.array([thrust, roll, pitch, yaw_cmd], dtype=np.float32)


# ---------------------------------------------------------------------------
# Procedural Trajectory Generation
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Generates 3D waypoints within arena boundary limits."""
    def __init__(self, bounds_x=(-2.0, 2.0), bounds_y=(-2.0, 2.0), bounds_z=(0.5, 2.0)):
        self.bx = bounds_x
        self.by = bounds_y
        self.bz = bounds_z

    def generate_waypoints(self, num_points: int = 6) -> np.ndarray:
        pts = np.zeros((num_points, 3))
        pts[0] = np.array([0.0, 0.0, 1.0])  # Start near center hover
        for i in range(1, num_points):
            pts[i] = [
                np.random.uniform(*self.bx),
                np.random.uniform(*self.by),
                np.random.uniform(*self.bz),
            ]
        return pts


def extract_kinematics(obs, info=None) -> np.ndarray:
    """Robustly extracts the 20-dim kinematic array whether env returns dict or array."""
    if isinstance(obs, dict):
        if "kin" in obs:
            return obs["kin"]
        elif "0" in obs:  # BaseAviary multi-agent dict fallback
            return obs["0"]
    return obs


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
    print(f"[COLLECT] Target episodes: {num_episodes} | Max steps/ep: {max_steps}")
    print(f"[COLLECT] Saving to: {zarr_path}")

    if os.path.exists(zarr_path):
        confirm = input(f"Zarr store '{zarr_path}' already exists. Overwrite? (y/n): ")
        if confirm.lower() == "y":
            shutil.rmtree(zarr_path)
        else:
            print("Aborting.")
            return

    # 1. Initialize Headless Environment & Native PID Controller
    env = gym.make(
        cfg.env.id,
        gui=False,
        obs=cfg.env.obs_type,
        act="pid",
        enable_ipc=False,
    )

    pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    traj_gen = TrajectoryGenerator()

    # In-memory episode buffers
    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_ends: List[int] = []

    cumulative_steps = 0
    dt = 1.0 / cfg.env.physics_hz

    # 2. Episode Rollout Loop
    for ep in tqdm(range(num_episodes), desc="Collecting Expert Flights"):
        obs_data, info = env.reset()
        current_kin = extract_kinematics(obs_data, info)

        waypoints = traj_gen.generate_waypoints(num_points=8)
        current_wp_idx = 1
        target_pos = waypoints[current_wp_idx]

        ep_obs = []
        ep_act = []

        for step in range(max_steps):
            # Parse current physical drone state from kinematics vector
            pos = current_kin[0:3]
            quat = current_kin[3:7]
            vel = current_kin[10:13]
            ang_vel = current_kin[13:16]

            # Advance waypoint if drone is within 25 cm of target
            dx = target_pos[0] - pos[0]
            dy = target_pos[1] - pos[1]
            dist_horizontal = np.hypot(dx, dy)
            dist_to_wp = np.linalg.norm(target_pos - pos)
            
            if dist_to_wp < 0.25 and current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                target_pos = waypoints[current_wp_idx]
                dx = target_pos[0] - pos[0]
                dy = target_pos[1] - pos[1]
                dist_horizontal = np.hypot(dx, dy)

            # Precompute desired yaw to turn camera toward target
            desired_yaw = np.arctan2(dy, dx) if dist_horizontal > 0.15 else quat_to_yaw(quat)

            # Compute analytic expert control using PID (commanding attitude & heading)
            computed_rpms, _, _ = pid_controller.computeControlFromState(
                control_timestep=dt,
                state=np.hstack([pos, quat, vel, ang_vel, np.zeros(4)]),
                target_pos=target_pos,
                target_rpy=np.array([0.0, 0.0, desired_yaw]), # <-- Instructs PID to rotate nose
                target_vel=np.zeros(3),
            )

            # Map PID outputs to normalized 4-channel action with active yaw rate
            action = compute_camera_aligned_action(
                pos=pos,
                quat=quat,
                target_pos=target_pos,
                computed_motor_rpms=computed_rpms,
            )

            # Handle Observation Storage 
            # (Assuming purely kinematic training for now. If using RGB, append obs_data["rgb"] to a separate list here).
            ep_obs.append(current_kin[:obs_dim].astype(np.float32))
            ep_act.append(action)

            # Step PyBullet simulation
            action_payload = {"0": action} if isinstance(obs_data, dict) and "0" in obs_data else action
            obs_data, reward, terminated, truncated, info = env.step(action_payload)
            current_kin = extract_kinematics(obs_data, info)

            if terminated or truncated:
                break

        # Discard degenerate episodes (crashed within first 50 steps)
        if len(ep_obs) > 50:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_act)
            cumulative_steps += len(ep_obs)
            episode_ends.append(cumulative_steps)

    env.close()

    # 3. Serialize to Chunked Zarr Storage
    print(f"\n[WRITE] Writing {cumulative_steps} steps to Zarr on disk...")
    root = zarr.open(zarr_path, mode="w")

    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Save arrays with temporal chunking (chunk size of 1000 steps optimizes SSD read access)
    data_group.create_dataset(
        "obs",
        data=np.array(all_obs, dtype=np.float32),
        chunks=(1000, obs_dim),
        dtype="float32",
    )
    data_group.create_dataset(
        "action",
        data=np.array(all_actions, dtype=np.float32),
        chunks=(1000, action_dim),
        dtype="float32",
    )
    meta_group.create_dataset(
        "episode_ends",
        data=np.array(episode_ends, dtype=np.int64),
        chunks=(1000,),
        dtype="int64",
    )

    print(f"[DONE] Dataset compiled successfully at '{zarr_path}'.")
    print(f"Total Trajectory Steps: {cumulative_steps}")
    print(f"Total Valid Episodes:   {len(episode_ends)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    collect_expert_trajectories(cfg)