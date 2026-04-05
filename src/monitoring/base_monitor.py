"""Shared infrastructure for live-plotting Cassie simulation monitors."""

from abc import ABC, abstractmethod
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from matplotlib.animation import FuncAnimation

from src.env.cassie import CassieEnv

# Common plotting style
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_RCPARAMS = {
    "figure.figsize": (24, 12),
    "figure.dpi": 120,
    "lines.linewidth": 1.5,
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
}

# Observation structure (indices based on CassieEnv._set_obs)
OBS_SLICES = {
    "actuatorpos": slice(0, 10),
    "jointpos": slice(10, 16),
    "framequat": slice(16, 20),
    "gyro": slice(20, 23),
    "accelerometer": slice(23, 26),
    "magnetometer": slice(26, 29),
    "command": slice(29, 31),
    "contact_forces": slice(31, 37),
    "clock": slice(37, 39),
}

OBS_COMPONENT_NAMES = {
    "actuatorpos": [
        "left-hip-roll",
        "left-hip-yaw",
        "left-hip-pitch",
        "left-knee",
        "left-foot",
        "right-hip-roll",
        "right-hip-yaw",
        "right-hip-pitch",
        "right-knee",
        "right-foot",
    ],
    "jointpos": [
        "left-shin",
        "left-tarsus",
        "left-foot-output",
        "right-shin",
        "right-tarsus",
        "right-foot-output",
    ],
    "framequat": ["w", "x", "y", "z"],
    "gyro": ["x", "y", "z"],
    "accelerometer": ["x", "y", "z"],
    "magnetometer": ["x", "y", "z"],
    "command": ["x_vel", "y_vel"],
    "contact_forces": ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
    "clock": ["sin", "cos"],
}


def apply_plot_style():
    """Apply the common plotting style."""
    plt.style.use(PLOT_STYLE)
    for k, v in PLOT_RCPARAMS.items():
        plt.rcParams[k] = v


def create_lines(ax, keys, colors, labels=None, key_suffix=False):
    """Create matplotlib line objects for an axis and return a dict of key -> Line2D."""
    if labels is None:
        labels = keys
    lines = {}
    for i, key in enumerate(keys):
        label = labels[i].split("_")[-1] if key_suffix else labels[i]
        (lines[key],) = ax.plot([], [], label=label, color=colors[i])
    return lines


class BaseMonitor(ABC):
    """Abstract base class for live Cassie simulation monitors."""

    def __init__(self, env_config: dict, max_data_points: int = 300):
        apply_plot_style()

        self.env = CassieEnv(env_config=env_config)
        self.max_data_points = max_data_points

        # Get body IDs from model
        self.pelvis_id = mj.mj_name2id(
            self.env.model, mj.mjtObj.mjOBJ_BODY, "cassie-pelvis"
        )
        self.left_foot_id = mj.mj_name2id(
            self.env.model, mj.mjtObj.mjOBJ_BODY, "left-foot"
        )
        self.right_foot_id = mj.mj_name2id(
            self.env.model, mj.mjtObj.mjOBJ_BODY, "right-foot"
        )

        self.data: Dict[str, list] = {"timestep": []}
        self.lines: Dict[str, plt.Line2D] = {}
        self.all_axes: List[plt.Axes] = []

        self.observation, _ = self.env.reset()
        self.timestep = 0
        self.last_render = np.zeros(
            (self.env.height, self.env.width, 3), dtype=np.uint8
        )
        self.terminated = False
        self.truncated = False

        # To be set by subclasses
        self.fig = None
        self.img_display = None

    def _init_data_keys(self, keys: List[str]):
        """Initialize data storage for additional keys."""
        for key in keys:
            if key not in self.data:
                self.data[key] = []

    def _append_and_trim(self, key: str, value):
        """Append a value to a data series, trimming to max_data_points."""
        self.data[key].append(value)

    def _trim_all(self):
        """Trim all data series to max_data_points."""
        if len(self.data["timestep"]) > self.max_data_points:
            for key in self.data:
                self.data[key].pop(0)

    def _store_context_data(self):
        """Store common context data (heights, positions)."""
        self.data["pelvis_height"].append(self.env.data.xpos[self.pelvis_id, 2])
        self.data["left_foot_height"].append(self.env.data.xpos[self.left_foot_id, 2])
        self.data["right_foot_height"].append(self.env.data.xpos[self.right_foot_id, 2])

    def _store_rpy(self, quaternion: np.ndarray):
        """Store pelvis RPY from quaternion."""
        norm = np.linalg.norm(quaternion)
        if norm > 1e-6:
            quaternion = quaternion / norm
        else:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        rpy = self.env.quat_to_rpy(quaternion, radians=False)
        self.data["pelvis_roll"].append(rpy[0])
        self.data["pelvis_pitch"].append(rpy[1])
        self.data["pelvis_yaw"].append(rpy[2])

    def _update_render(self):
        """Get render frame from environment."""
        render_result = self.env.render()
        if render_result is not None and isinstance(render_result, np.ndarray):
            if render_result.shape[0:2] == (self.env.height, self.env.width):
                self.last_render = render_result
            else:
                self.last_render = cv2.resize(
                    render_result, (self.env.width, self.env.height)
                )

    def _finalize_axes(self):
        """Apply common formatting to all axes."""
        for ax in self.all_axes:
            ax.set_xlim(0, self.max_data_points)
            ax.grid(True, linestyle="--", alpha=0.6)
            if ax.get_lines():
                ax.legend(loc="upper right")

    @abstractmethod
    def setup_figure(self):
        """Setup matplotlib figure and axes. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def update_data(self):
        """Collect one step of data. Must be implemented by subclasses."""
        ...

    def update_plot(self, frame):
        """Animation callback: update data and redraw."""
        try:
            self.update_data()
        except Exception as e:
            print(f"Error during data update: {e}")
            return []

        artists = []

        if self.img_display is not None:
            self.img_display.set_array(self.last_render)
            artists.append(self.img_display)

        timesteps = self.data["timestep"]
        if not timesteps:
            return artists

        for key, line in self.lines.items():
            if key in self.data and len(self.data[key]) == len(timesteps):
                line.set_data(timesteps, self.data[key])
                artists.append(line)

        min_time = timesteps[0]
        max_time = max(timesteps[-1], min_time + 1)

        for ax in self.all_axes:
            ax.set_xlim(min_time, max_time)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        return artists

    def run(
        self, title: str = "Live Cassie Monitor", frames: int = 1000, interval: int = 40
    ):
        """Start the live animation."""
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=frames,
            interval=interval,
            blit=True,
            repeat=False,
        )
        self.fig.suptitle(title, fontsize=14)
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")
        finally:
            print("Closing environment.")
            self.env.close()
