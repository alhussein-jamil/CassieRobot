"""Live monitoring dashboard for Cassie simulation - observation-focused view."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.monitoring.base_monitor import (
    BaseMonitor,
    OBS_COMPONENT_NAMES,
    OBS_SLICES,
    create_lines,
)


class LiveCassieMonitor(BaseMonitor):
    def __init__(self, max_data_points=300):
        super().__init__(
            env_config={
                "symmetric_regulation": "none",
                "sim_fps": 40,
                "push_prob": 0.01,
                "push_duration": 0.1,
            },
            max_data_points=max_data_points,
        )

        # Initialize observation data keys
        for category, names in OBS_COMPONENT_NAMES.items():
            for name in names:
                self.data[f"{category}_{name}"] = []

        # Context + reward data keys
        extra_keys = [
            "pelvis_height",
            "left_foot_height",
            "right_foot_height",
            "pelvis_roll",
            "pelvis_pitch",
            "pelvis_yaw",
            "feet_distance_x",
            "feet_distance_y",
            "total_reward",
            "r_biped",
            "r_cmd",
            "r_smooth",
            "r_feet_parallel",
            "c_frc_left",
            "c_frc_right",
            "c_spd_left",
            "c_spd_right",
            "exp_q_left_frc",
            "exp_q_right_frc",
            "exp_q_left_spd",
            "exp_q_right_spd",
        ]
        self._init_data_keys(extra_keys)

        self.setup_figure()

    def setup_figure(self):
        self.fig = plt.figure(figsize=plt.rcParams["figure.figsize"])
        gs = GridSpec(3, 4, figure=self.fig, width_ratios=[1.5, 1, 1, 1])

        # Video
        ax_video = self.fig.add_subplot(gs[:, 0])
        ax_video.set_title("Cassie Simulation")
        ax_video.axis("off")
        self.img_display = ax_video.imshow(
            np.zeros((self.env.height, self.env.width, 3), dtype=np.uint8)
        )

        # Row 0
        ax_motors_left = self.fig.add_subplot(gs[0, 1])
        ax_motors_left.set_title("Left Leg Motors")
        ax_motors_left.set_ylabel("Position (deg)")
        ax_motors_left.tick_params(axis="x", labelbottom=False)

        ax_imu = self.fig.add_subplot(gs[0, 2])
        ax_imu.set_title("Pelvis Orientation (RPY)")
        ax_imu.set_ylabel("Angle (deg)")
        ax_imu.tick_params(axis="x", labelbottom=False)

        ax_heights = self.fig.add_subplot(gs[0, 3])
        ax_heights.set_title("Heights")
        ax_heights.set_ylabel("Height (m)")
        ax_heights.tick_params(axis="x", labelbottom=False)

        # Row 1
        ax_motors_right = self.fig.add_subplot(gs[1, 1])
        ax_motors_right.set_title("Right Leg Motors")
        ax_motors_right.set_ylabel("Position (deg)")
        ax_motors_right.tick_params(axis="x", labelbottom=False)

        ax_exp = self.fig.add_subplot(gs[1, 2])
        ax_exp.set_title("Exp(Reward Components)")
        ax_exp.set_ylabel("Value (Post-Exp)")
        ax_exp.tick_params(axis="x", labelbottom=False)

        ax_forces = self.fig.add_subplot(gs[1, 3])
        ax_forces.set_title("Contact Forces (World Frame)")
        ax_forces.set_ylabel("Force (N)")
        ax_forces.tick_params(axis="x", labelbottom=False)

        # Row 2
        ax_joints = self.fig.add_subplot(gs[2, 1])
        ax_joints.set_title("Joint Positions")
        ax_joints.set_ylabel("Position (deg)")
        ax_joints.set_xlabel("Timestep")

        ax_coeffs = self.fig.add_subplot(gs[2, 2])
        ax_coeffs.set_title("Reward Coefficients")
        ax_coeffs.set_ylabel("Coefficient Value")
        ax_coeffs.set_xlabel("Timestep")

        ax_rewards = self.fig.add_subplot(gs[2, 3])
        ax_rewards.set_title("Rewards")
        ax_rewards.set_ylabel("Reward Value")
        ax_rewards.set_xlabel("Timestep")

        self.all_axes = [
            ax_motors_left,
            ax_imu,
            ax_heights,
            ax_motors_right,
            ax_exp,
            ax_forces,
            ax_joints,
            ax_coeffs,
            ax_rewards,
        ]

        # Create lines
        left_keys = [f"actuatorpos_{n}" for n in OBS_COMPONENT_NAMES["actuatorpos"][:5]]
        self.lines.update(
            create_lines(
                ax_motors_left,
                left_keys,
                sns.color_palette("tab10", 5),
                key_suffix=True,
            )
        )

        self.lines.update(
            create_lines(
                ax_imu,
                ["pelvis_roll", "pelvis_pitch", "pelvis_yaw"],
                sns.color_palette("Accent", 3),
            )
        )

        self.lines.update(
            create_lines(
                ax_heights,
                ["pelvis_height", "left_foot_height", "right_foot_height"],
                sns.color_palette("Paired", 3),
            )
        )

        right_keys = [
            f"actuatorpos_{n}" for n in OBS_COMPONENT_NAMES["actuatorpos"][5:]
        ]
        self.lines.update(
            create_lines(
                ax_motors_right,
                right_keys,
                sns.color_palette("Set2", 5),
                key_suffix=True,
            )
        )

        self.lines.update(
            create_lines(
                ax_exp,
                [
                    "exp_q_left_frc",
                    "exp_q_right_frc",
                    "exp_q_left_spd",
                    "exp_q_right_spd",
                ],
                sns.color_palette("Dark2", 4),
                labels=["Exp_Frc_L", "Exp_Frc_R", "Exp_Spd_L", "Exp_Spd_R"],
            )
        )

        force_keys = [
            f"contact_forces_{n}" for n in OBS_COMPONENT_NAMES["contact_forces"]
        ]
        self.lines.update(
            create_lines(
                ax_forces,
                force_keys,
                sns.color_palette("tab10", 6),
                labels=["L_Fx", "L_Fy", "L_Fz", "R_Fx", "R_Fy", "R_Fz"],
            )
        )

        joint_keys = [f"jointpos_{n}" for n in OBS_COMPONENT_NAMES["jointpos"]]
        self.lines.update(
            create_lines(
                ax_joints, joint_keys, sns.color_palette("Paired", 6), key_suffix=True
            )
        )

        self.lines.update(
            create_lines(
                ax_coeffs,
                ["c_frc_left", "c_frc_right", "c_spd_left", "c_spd_right"],
                sns.color_palette("Set1", 4),
                labels=["Frc_L", "Frc_R", "Spd_L", "Spd_R"],
            )
        )

        self.lines.update(
            create_lines(
                ax_rewards,
                ["total_reward", "r_biped", "r_cmd", "r_smooth", "r_feet_parallel"],
                sns.color_palette("Dark2", 5),
                labels=["Total", "Biped", "Cmd", "Smooth", "Feet Parallel"],
            )
        )

        self._finalize_axes()
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.subplots_adjust(hspace=0.2, wspace=0.25)

    def update_data(self):
        action = np.random.uniform(-0.1, 0.1, self.env.action_space.shape[0])
        self.observation, reward, self.terminated, self.truncated, info = self.env.step(
            action
        )

        self._update_render()
        self._trim_all()
        self.data["timestep"].append(self.timestep)

        # Context data
        self._store_context_data()

        left_foot_pos = self.env.data.xpos[self.left_foot_id]
        right_foot_pos = self.env.data.xpos[self.right_foot_id]
        self.data["feet_distance_x"].append(abs(left_foot_pos[0] - right_foot_pos[0]))
        self.data["feet_distance_y"].append(abs(left_foot_pos[1] - right_foot_pos[1]))

        # Parse observation
        for category, obs_slice in OBS_SLICES.items():
            names = OBS_COMPONENT_NAMES[category]
            values = self.observation[obs_slice]
            for i, name in enumerate(names):
                val = (
                    values[i]
                    if isinstance(values, np.ndarray) and values.ndim > 0
                    else values
                )
                self.data[f"{category}_{name}"].append(val)

        # RPY
        pelvis_quat = self.observation[OBS_SLICES["framequat"]]
        self._store_rpy(pelvis_quat)

        # Rewards / coefficients / exp components
        cm = info.get("custom_metrics", {})
        self.data["total_reward"].append(reward)
        for key in [
            "r_biped",
            "r_cmd",
            "r_smooth",
            "r_feet_parallel",
            "c_frc_left",
            "c_frc_right",
            "c_spd_left",
            "c_spd_right",
            "exp_q_left_frc",
            "exp_q_right_frc",
            "exp_q_left_spd",
            "exp_q_right_spd",
        ]:
            self.data[key].append(cm.get(key, np.nan))

        self.timestep += 1


if __name__ == "__main__":
    monitor = LiveCassieMonitor(max_data_points=300)
    monitor.run(title="Live Cassie Robot Monitor", frames=150, interval=40)
