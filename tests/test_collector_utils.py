# tests/test_collector_utils.py

import math
import numpy as np
import pytest

from dataset.collect_expert import (
    process_fpv_image,
    extract_drone_obs,
    quat_to_yaw,
    compute_camera_aligned_action,
    get_obstacle_free_carrot,
    TrajectoryGenerator,
)


class TestCollectorUtilities:
    def test_process_fpv_image(self):
        # PyBullet RGBA format: [H, W, 4]
        raw_rgba = np.random.randint(0, 255, size=(100, 100, 4), dtype=np.uint8)
        processed = process_fpv_image(raw_rgba, target_size=(224, 224))

        # Expected PyTorch image format: [3, 224, 224]
        assert processed.shape == (3, 224, 224)
        assert processed.dtype == np.uint8

    def test_extract_drone_obs(self):
        # Single drone dictionary key
        obs_dict = {"0": np.array([1.0, 2.0, 3.0])}
        assert np.array_equal(extract_drone_obs(obs_dict), np.array([1.0, 2.0, 3.0]))

        # Plain array fallback
        raw_arr = np.array([4.0, 5.0, 6.0])
        assert np.array_equal(extract_drone_obs(raw_arr), raw_arr)

    def test_quat_to_yaw(self):
        # Identity quaternion [qx=0, qy=0, qz=0, qw=1] -> 0 radians
        yaw_identity = quat_to_yaw(np.array([0.0, 0.0, 0.0, 1.0]))
        assert math.isclose(yaw_identity, 0.0, abs_tol=1e-5)

        # 90 degrees yaw rotation around Z axis: [0, 0, sin(pi/4), cos(pi/4)]
        half_angle = math.pi / 4.0
        quat_90 = np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])
        yaw_90 = quat_to_yaw(quat_90)
        assert math.isclose(yaw_90, math.pi / 2.0, abs_tol=1e-5)

    def test_compute_camera_aligned_action(self):
        pos = np.array([0.0, 0.0, 1.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([1.0, 1.0, 1.5])
        rpms = np.array([14000.0, 14000.0, 14000.0, 14000.0])

        action = compute_camera_aligned_action(
            pos=pos,
            quat=quat,
            target_pos=target_pos,
            computed_motor_rpms=rpms,
            kp_yaw=1.8,
        )

        assert action.shape == (4,)
        assert action.dtype == np.float32
        # All actions must be clipped to normalized limits [-1.0, 1.0]
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_trajectory_generator(self):
        bx, by, bz = (-2.0, 2.0), (-2.0, 2.0), (0.5, 2.0)
        gen = TrajectoryGenerator(bounds_x=bx, bounds_y=by, bounds_z=bz)

        num_points = 8
        pts = gen.generate_waypoints(num_points=num_points)

        assert pts.shape == (num_points, 3)
        # First point is always the origin hover point [0, 0, 1]
        assert np.allclose(pts[0], np.array([0.0, 0.0, 1.0]))

        # Subsequent points must be within bounds
        for i in range(1, num_points):
            assert bx[0] <= pts[i, 0] <= bx[1]
            assert by[0] <= pts[i, 1] <= by[1]
            assert bz[0] <= pts[i, 2] <= bz[1]

    def test_get_obstacle_free_carrot_close_target(self):
        pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([0.05, 0.0, 1.0])
        carrot = get_obstacle_free_carrot(pos, target_pos, client_id=0)
        assert np.allclose(carrot, target_pos)
