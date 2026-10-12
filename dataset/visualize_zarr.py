# dataset/visualize_zarr.py
"""
Interactive Matplotlib visualizer for drone flight trajectories stored in Zarr format.

Features:
- FPV camera RGB frame display.
- Previous / Next buttons to step through frames.
- Interactive slider to scrub to any timestep.
- Clean telemetry panel displaying:
  * Drone position (x, y, z) and target goal (gx, gy, gz)
  * Distance to goal
  * Orientation (quaternion & roll/pitch/yaw in degrees)
  * Linear velocity and speed
  * Angular rates
  * Motor RPMs
  * Expert control actions (Thrust, Roll, Pitch, Yaw) with visual bar indicators
- Keyboard shortcuts:
  * Left / Right arrows: Previous / Next step
  * Up / Down arrows: Next / Previous episode
  * Space: Auto-play / Pause
"""

import os
import argparse
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


class TrajectoryVisualizer:
    def __init__(self, zarr_path: str = "data/expert_trajectories.zarr", start_episode: int = 0):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found at: {zarr_path}")

        self.zarr_path = zarr_path
        self.root = zarr.open(zarr_path, mode="r")

        # Validate dataset structure
        if "data/img" not in self.root or "data/obs" not in self.root or "data/action" not in self.root:
            raise KeyError(f"Expected groups 'data/img', 'data/obs', 'data/action' inside {zarr_path}")

        self.img_data = self.root["data/img"]
        self.obs_data = self.root["data/obs"]
        self.act_data = self.root["data/action"]

        # Parse episode boundaries
        if "meta/episode_ends" in self.root:
            raw_ends = self.root["meta/episode_ends"][:]
            self.episodes = []
            prev_end = 0
            for ep_end in raw_ends:
                self.episodes.append((prev_end, int(ep_end)))
                prev_end = int(ep_end)
        else:
            total_steps = len(self.obs_data)
            self.episodes = [(0, total_steps)]

        self.num_episodes = len(self.episodes)
        self.current_ep = max(0, min(start_episode, self.num_episodes - 1))
        self.current_step = 0

        # Animation state
        self.is_playing = False
        self.anim_timer = None

        print(f"[LOAD] Opened Zarr store '{zarr_path}'")
        print(f"[INFO] Total Episodes: {self.num_episodes} | Total Steps: {len(self.obs_data)}")
        self._setup_figure()
        self.update_display()

    def _setup_figure(self):
        # Create main window layout
        plt.style.use("fast")
        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title(f"Drone Trajectory Viewer - {os.path.basename(self.zarr_path)}")

        # Grid layout:
        # Left: FPV Camera Frame [0.06, 0.22, 0.42, 0.68]
        # Right: Telemetry & Actions [0.52, 0.22, 0.44, 0.68]
        self.ax_img = self.fig.add_axes([0.06, 0.22, 0.42, 0.68])
        self.ax_info = self.fig.add_axes([0.52, 0.22, 0.44, 0.68])

        # Bottom navigation controls
        # Slider
        self.ax_slider = self.fig.add_axes([0.15, 0.10, 0.45, 0.035])
        ep_start, ep_end = self.episodes[self.current_ep]
        max_step = max(1, ep_end - ep_start - 1)
        self.slider = Slider(
            self.ax_slider,
            "Step",
            0,
            max_step,
            valinit=0,
            valstep=1,
            valfmt="%d",
            color="#2a75d3",
        )
        self.slider.on_changed(self._on_slider_changed)

        # Buttons: Previous & Next
        self.ax_prev = self.fig.add_axes([0.62, 0.09, 0.10, 0.055])
        self.btn_prev = Button(self.ax_prev, "◀ Previous", color="#e0e0e0", hovercolor="#c8d6e5")
        self.btn_prev.on_clicked(self._on_prev_clicked)

        self.ax_next = self.fig.add_axes([0.73, 0.09, 0.10, 0.055])
        self.btn_next = Button(self.ax_next, "Next ▶", color="#2a75d3", hovercolor="#1e5ba8")
        self.btn_next.label.set_color("white")
        self.btn_next.label.set_weight("bold")
        self.btn_next.on_clicked(self._on_next_clicked)

        # Play/Pause button
        self.ax_play = self.fig.add_axes([0.84, 0.09, 0.10, 0.055])
        self.btn_play = Button(self.ax_play, "▶ Play", color="#10ac84", hovercolor="#0e9774")
        self.btn_play.label.set_color("white")
        self.btn_play.on_clicked(self._on_play_clicked)

        # Episode Previous / Next buttons
        self.ax_ep_prev = self.fig.add_axes([0.02, 0.09, 0.05, 0.055])
        self.btn_ep_prev = Button(self.ax_ep_prev, "◀ Ep", color="#dfe4ea", hovercolor="#ced6e0")
        self.btn_ep_prev.on_clicked(self._on_ep_prev_clicked)

        self.ax_ep_next = self.fig.add_axes([0.08, 0.09, 0.05, 0.055])
        self.btn_ep_next = Button(self.ax_ep_next, "Ep ▶", color="#dfe4ea", hovercolor="#ced6e0")
        self.btn_ep_next.on_clicked(self._on_ep_next_clicked)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Initial image placeholder
        self.im_display = None

    def _get_current_global_index(self) -> int:
        ep_start, ep_end = self.episodes[self.current_ep]
        return ep_start + self.current_step

    def _get_current_episode_len(self) -> int:
        ep_start, ep_end = self.episodes[self.current_ep]
        return max(1, ep_end - ep_start)

    def _on_slider_changed(self, val):
        self.current_step = int(val)
        self.update_display(update_slider=False)

    def _on_prev_clicked(self, event=None):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
        elif self.current_ep > 0:
            # Wrap to previous episode end
            self.current_ep -= 1
            self._update_slider_range()
            self.current_step = self._get_current_episode_len() - 1
            self.update_display()

    def _on_next_clicked(self, event=None):
        ep_len = self._get_current_episode_len()
        if self.current_step < ep_len - 1:
            self.current_step += 1
            self.update_display()
        elif self.current_ep < self.num_episodes - 1:
            # Advance to next episode start
            self.current_ep += 1
            self._update_slider_range()
            self.current_step = 0
            self.update_display()
        else:
            if self.is_playing:
                self._stop_playback()

    def _on_ep_prev_clicked(self, event=None):
        if self.current_ep > 0:
            self.current_ep -= 1
            self._update_slider_range()
            self.current_step = 0
            self.update_display()

    def _on_ep_next_clicked(self, event=None):
        if self.current_ep < self.num_episodes - 1:
            self.current_ep += 1
            self._update_slider_range()
            self.current_step = 0
            self.update_display()

    def _on_play_clicked(self, event=None):
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self.is_playing = True
        self.btn_play.label.set_text("⏸ Pause")
        self.btn_play.color = "#ee5253"
        self.btn_play.hovercolor = "#d63031"
        self.fig.canvas.draw_idle()
        if self.anim_timer is None:
            self.anim_timer = self.fig.canvas.new_timer(interval=50)  # 20 Hz (50ms)
            self.anim_timer.add_callback(self._on_timer_tick)
        self.anim_timer.start()

    def _stop_playback(self):
        self.is_playing = False
        self.btn_play.label.set_text("▶ Play")
        self.btn_play.color = "#10ac84"
        self.btn_play.hovercolor = "#0e9774"
        self.fig.canvas.draw_idle()
        if self.anim_timer is not None:
            self.anim_timer.stop()

    def _on_timer_tick(self):
        if not self.is_playing:
            return
        ep_len = self._get_current_episode_len()
        if self.current_step < ep_len - 1:
            self.current_step += 1
            self.update_display()
        else:
            if self.current_ep < self.num_episodes - 1:
                self.current_ep += 1
                self._update_slider_range()
                self.current_step = 0
                self.update_display()
            else:
                self._stop_playback()

    def _on_key_press(self, event):
        if event.key in ["right", "d", "n"]:
            self._on_next_clicked()
        elif event.key in ["left", "a", "p"]:
            self._on_prev_clicked()
        elif event.key in ["up", "w"]:
            self._on_ep_next_clicked()
        elif event.key in ["down", "s"]:
            self._on_ep_prev_clicked()
        elif event.key == " ":
            self._on_play_clicked()

    def _update_slider_range(self):
        ep_len = self._get_current_episode_len()
        self.slider.valmax = max(1, ep_len - 1)
        self.ax_slider.set_xlim(0, max(1, ep_len - 1))
        self.slider.set_val(0)

    def _format_action_bar(self, val: float, length: int = 14) -> str:
        """Renders an ASCII bi-directional bar chart for values between -1.0 and 1.0."""
        val = float(np.clip(val, -1.0, 1.0))
        half = length // 2
        norm = int(round(abs(val) * half))
        if val >= 0:
            bar = " " * half + "|" + "#" * norm + " " * (half - norm)
        else:
            bar = " " * (half - norm) + "#" * norm + "|" + " " * half
        return f"[{bar}] {val:+0.2f}"

    def update_display(self, update_slider: bool = True):
        global_idx = self._get_current_global_index()
        ep_len = self._get_current_episode_len()

        # Update slider position if triggered externally
        if update_slider:
            self.slider.eventson = False
            self.slider.set_val(self.current_step)
            self.slider.eventson = True

        # 1. Fetch data for current timestep
        raw_img = self.img_data[global_idx]  # Shape [3, H, W]
        if raw_img.ndim == 3 and raw_img.shape[0] in [1, 3, 4]:
            # Convert channel-first [3, H, W] to channel-last [H, W, 3]
            img_rgb = np.transpose(raw_img, (1, 2, 0))
        else:
            img_rgb = raw_img

        raw_obs = self.obs_data[global_idx]
        raw_act = self.act_data[global_idx]

        # 2. Parse Observation & Action Components
        # [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), rpm(4), goal(3)]
        pos = raw_obs[0:3]
        quat = raw_obs[3:7] if len(raw_obs) >= 7 else np.zeros(4)
        rpy_rad = raw_obs[7:10] if len(raw_obs) >= 10 else np.zeros(3)
        rpy_deg = np.degrees(rpy_rad)
        vel = raw_obs[10:13] if len(raw_obs) >= 13 else np.zeros(3)
        speed = float(np.linalg.norm(vel))
        ang_vel = raw_obs[13:16] if len(raw_obs) >= 16 else np.zeros(3)
        motor_rpms = raw_obs[16:20] if len(raw_obs) >= 20 else np.zeros(4)
        goal = raw_obs[20:23] if len(raw_obs) >= 23 else np.zeros(3)
        dist_to_goal = float(np.linalg.norm(goal - pos))

        thrust_cmd = raw_act[0] if len(raw_act) > 0 else 0.0
        roll_cmd = raw_act[1] if len(raw_act) > 1 else 0.0
        pitch_cmd = raw_act[2] if len(raw_act) > 2 else 0.0
        yaw_cmd = raw_act[3] if len(raw_act) > 3 else 0.0

        # 3. Render FPV Camera Image with on-screen HUD overlay
        self.ax_img.clear()
        self.ax_img.imshow(img_rgb)
        self.ax_img.set_title(
            f"FPV Camera | Step {self.current_step + 1}/{ep_len} (Frame #{global_idx})",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        self.ax_img.axis("off")

        # Top overlay: Alt, Speed, Distance to Goal
        hud_top = f"Alt: {pos[2]:.2f}m | Spd: {speed:.2f}m/s | Goal Dist: {dist_to_goal:.2f}m"
        self.ax_img.text(
            0.03,
            0.95,
            hud_top,
            transform=self.ax_img.transAxes,
            fontsize=8.5,
            family="monospace",
            color="white",
            weight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e272e", alpha=0.75, edgecolor="none"),
        )

        # Bottom overlay: Control Actions
        hud_act = f"Thr: {thrust_cmd:+.2f} | R: {roll_cmd:+.2f} | P: {pitch_cmd:+.2f} | Y: {yaw_cmd:+.2f}"
        self.ax_img.text(
            0.03,
            0.05,
            hud_act,
            transform=self.ax_img.transAxes,
            fontsize=8.5,
            family="monospace",
            color="#0be881",
            weight="bold",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e272e", alpha=0.75, edgecolor="none"),
        )

        # 5. Format Information Card Text
        self.ax_info.clear()
        self.ax_info.axis("off")

        info_text = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  FLIGHT TELEMETRY & CONTROLS\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  • Episode:         {self.current_ep + 1} / {self.num_episodes}\n"
            f"  • Episode Step:    {self.current_step + 1} / {ep_len}\n"
            f"  • Global Index:    {global_idx} / {len(self.obs_data)}\n"
            f"\n"
            f"【 3D POSITION & NAVIGATION 】\n"
            f"  • Current (x,y,z): [ {pos[0]:+6.2f}, {pos[1]:+6.2f}, {pos[2]:+6.2f} ] m\n"
            f"  • Target  (x,y,z): [ {goal[0]:+6.2f}, {goal[1]:+6.2f}, {goal[2]:+6.2f} ] m\n"
            f"  • Dist to Goal:    {dist_to_goal:6.2f} m\n"
            f"  • Altitude:        {pos[2]:6.2f} m\n"
            f"\n"
            f"【 KINEMATICS & DYNAMICS 】\n"
            f"  • Linear Vel (v):  [ {vel[0]:+5.2f}, {vel[1]:+5.2f}, {vel[2]:+5.2f} ] m/s\n"
            f"  • Linear Speed:    {speed:5.2f} m/s\n"
            f"  • Euler RPY (deg): [ R:{rpy_deg[0]:+5.1f}°, P:{rpy_deg[1]:+5.1f}°, Y:{rpy_deg[2]:+5.1f}° ]\n"
            f"  • Angular Vel (w): [ {ang_vel[0]:+5.2f}, {ang_vel[1]:+5.2f}, {ang_vel[2]:+5.2f} ] rad/s\n"
            f"  • Motor RPMs:      [ {motor_rpms[0]:.0f}, {motor_rpms[1]:.0f}, {motor_rpms[2]:.0f}, {motor_rpms[3]:.0f} ]\n"
            f"\n"
            f"【 EXPERT ACTION COMMANDS 】\n"
            f"  • Thrust:  {self._format_action_bar(thrust_cmd)}\n"
            f"  • Roll:    {self._format_action_bar(roll_cmd)}\n"
            f"  • Pitch:   {self._format_action_bar(pitch_cmd)}\n"
            f"  • Yaw:     {self._format_action_bar(yaw_cmd)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f" Shortcuts: [Left/Right]=Step  [Up/Down]=Episode  [Space]=Play/Pause"
        )

        self.ax_info.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax_info.transAxes,
            fontsize=9.5,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced6e0", alpha=0.9),
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Drone Zarr Trajectory Interactive Visualizer")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="data/expert_trajectories.zarr",
        help="Path to the Zarr dataset store",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize (0-indexed)",
    )
    args = parser.parse_args()

    # Normalize path relative to project root if needed
    zarr_path = args.zarr_path
    if not os.path.isabs(zarr_path) and not os.path.exists(zarr_path):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(proj_root, zarr_path)
        if os.path.exists(candidate):
            zarr_path = candidate

    viz = TrajectoryVisualizer(zarr_path=zarr_path, start_episode=args.episode)
    viz.show()


if __name__ == "__main__":
    main()
