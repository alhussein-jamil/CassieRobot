"""Live monitoring dashboard for Cassie simulation - reward-focused view."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.env.constants import THETA_LEFT, THETA_RIGHT, c_stance_spd, c_swing_frc
from src.monitoring.base_monitor import BaseMonitor, create_lines


class RewardCassieMonitor(BaseMonitor):
    def __init__(self, max_data_points=400):
        super().__init__(
            env_config={"symmetric_regulation": "none", "sim_fps": 40},
            max_data_points=max_data_points,
        )

        # Initialize all data keys
        keys = [
            "pelvis_height",
            "left_foot_height",
            "right_foot_height",
            "pelvis_roll",
            "pelvis_pitch",
            "pelvis_yaw",
            # Raw
            "raw_q_vx",
            "raw_q_vy",
            "raw_q_left_frc",
            "raw_q_right_frc",
            "raw_q_left_spd",
            "raw_q_right_spd",
            "raw_q_action",
            "raw_q_pelvis_acc",
            "raw_q_torque",
            "raw_q_orientation",
            # Normalized
            "norm_q_vx",
            "norm_q_vy",
            "norm_q_left_frc",
            "norm_q_right_frc",
            "norm_q_left_spd",
            "norm_q_right_spd",
            "norm_q_action",
            "norm_q_pelvis_acc",
            "norm_q_torque",
            "norm_q_orientation",
            # Exponential (subset)
            "exp_q_pelvis_acc",
            "exp_q_vx",
            "exp_q_vy",
            "exp_q_left_frc",
            "exp_q_right_frc",
            "exp_q_action",
            "exp_q_orientation",
            "exp_q_torque",
            # Phase coefficients
            "c_frc_left",
            "c_frc_right",
            "c_spd_left",
            "c_spd_right",
            # Rewards
            "total_reward",
            "r_biped",
            "r_cmd",
            "r_smooth",
        ]
        self._init_data_keys(keys)

        self.setup_figure()

    def setup_figure(self):
        self.fig = plt.figure(figsize=plt.rcParams["figure.figsize"])
        gs = GridSpec(4, 4, figure=self.fig, width_ratios=[1.5, 1, 1, 1])

        # Video
        ax_video = self.fig.add_subplot(gs[0:3, 0])
        ax_video.set_title("Cassie Simulation")
        ax_video.axis("off")
        self.img_display = ax_video.imshow(
            np.zeros((self.env.height, self.env.width, 3), dtype=np.uint8)
        )

        # Row 0
        ax_raw_vel = self.fig.add_subplot(gs[0, 1])
        ax_raw_vel.set_title("Raw Pelvis Vel Err")
        ax_raw_vel.set_ylabel("Abs Vel Err (m/s)")
        ax_raw_vel.tick_params(axis="x", labelbottom=False)

        ax_raw_oa = self.fig.add_subplot(gs[0, 2])
        ax_raw_oa.set_title("Raw Orient Err & Pelvis Acc")
        ax_raw_oa.set_ylabel("Norm")
        ax_raw_oa.tick_params(axis="x", labelbottom=False)

        ax_heights = self.fig.add_subplot(gs[0, 3])
        ax_heights.set_title("Heights & Pelvis RPY")
        ax_heights.set_ylabel("Meters / Deg")
        ax_heights.tick_params(axis="x", labelbottom=False)

        # Row 1
        ax_raw_frc = self.fig.add_subplot(gs[1, 1])
        ax_raw_frc.set_title("Raw Foot Forces")
        ax_raw_frc.set_ylabel("Force Norm (N)")
        ax_raw_frc.tick_params(axis="x", labelbottom=False)

        ax_raw_spd = self.fig.add_subplot(gs[1, 2])
        ax_raw_spd.set_title("Raw Foot Speeds")
        ax_raw_spd.set_ylabel("Abs Speed")
        ax_raw_spd.tick_params(axis="x", labelbottom=False)

        ax_coeffs = self.fig.add_subplot(gs[1, 3])
        ax_coeffs.set_title("Phase Coefficients")
        ax_coeffs.set_ylabel("Value")
        ax_coeffs.tick_params(axis="x", labelbottom=False)

        # Row 2
        ax_exp = self.fig.add_subplot(gs[2, 1])
        ax_exp.set_title("Exponential Quantities")
        ax_exp.set_ylabel("Value (0-1)")
        ax_exp.set_xlabel("Timestep")

        ax_rew = self.fig.add_subplot(gs[2, 2])
        ax_rew.set_title("Reward Components")
        ax_rew.set_ylabel("Reward Value")
        ax_rew.set_xlabel("Timestep")

        ax_total = self.fig.add_subplot(gs[2, 3])
        ax_total.set_title("Total Reward")
        ax_total.set_ylabel("Reward Value")
        ax_total.tick_params(axis="x", labelbottom=False)

        # Row 3 (full width)
        ax_norm = self.fig.add_subplot(gs[3, :])
        ax_norm.set_title("Normalized Quantities (0-1)")
        ax_norm.set_ylabel("Norm Value")
        ax_norm.set_xlabel("Timestep")

        self.all_axes = [
            ax_raw_vel,
            ax_raw_oa,
            ax_heights,
            ax_raw_frc,
            ax_raw_spd,
            ax_coeffs,
            ax_exp,
            ax_rew,
            ax_total,
            ax_norm,
        ]

        # Lines
        self.lines.update(
            create_lines(
                ax_raw_vel,
                ["raw_q_vx", "raw_q_vy"],
                sns.color_palette("viridis", 2),
                labels=["q_vx", "q_vy"],
            )
        )

        self.lines.update(
            create_lines(
                ax_raw_oa,
                ["raw_q_orientation", "raw_q_pelvis_acc"],
                sns.color_palette("plasma", 2),
                labels=["q_orient", "q_pelvis_acc"],
            )
        )

        self.lines.update(
            create_lines(
                ax_heights,
                [
                    "pelvis_height",
                    "left_foot_height",
                    "right_foot_height",
                    "pelvis_roll",
                    "pelvis_pitch",
                    "pelvis_yaw",
                ],
                sns.color_palette("Paired", 6),
                labels=["Pelvis", "L Foot", "R Foot", "Roll", "Pitch", "Yaw"],
            )
        )

        self.lines.update(
            create_lines(
                ax_raw_frc,
                ["raw_q_left_frc", "raw_q_right_frc"],
                sns.color_palette("coolwarm", 2),
                labels=["q_L_frc", "q_R_frc"],
            )
        )

        self.lines.update(
            create_lines(
                ax_raw_spd,
                ["raw_q_left_spd", "raw_q_right_spd"],
                sns.color_palette("PRGn", 2),
                labels=["q_L_spd", "q_R_spd"],
            )
        )

        self.lines.update(
            create_lines(
                ax_coeffs,
                ["c_frc_left", "c_frc_right", "c_spd_left", "c_spd_right"],
                sns.color_palette("Spectral", 4),
                labels=["c_frc_L", "c_frc_R", "c_spd_L", "c_spd_R"],
            )
        )

        exp_keys = [
            "exp_q_vx",
            "exp_q_vy",
            "exp_q_left_frc",
            "exp_q_right_frc",
            "exp_q_action",
            "exp_q_orientation",
            "exp_q_pelvis_acc",
            "exp_q_torque",
        ]
        self.lines.update(
            create_lines(
                ax_exp,
                exp_keys,
                sns.color_palette("tab10", 8),
                labels=[
                    "vx",
                    "vy",
                    "L_frc",
                    "R_frc",
                    "action",
                    "orient",
                    "acc",
                    "torque",
                ],
            )
        )

        self.lines.update(
            create_lines(
                ax_rew,
                ["r_biped", "r_cmd", "r_smooth"],
                sns.color_palette("Accent", 3),
                labels=["Biped", "Cmd", "Smooth"],
            )
        )

        self.lines.update(
            create_lines(
                ax_total,
                ["total_reward"],
                sns.color_palette("Set1", 1),
                labels=["Total"],
            )
        )

        norm_keys = [
            "norm_q_vx",
            "norm_q_vy",
            "norm_q_orientation",
            "norm_q_pelvis_acc",
            "norm_q_left_frc",
            "norm_q_right_frc",
            "norm_q_left_spd",
            "norm_q_right_spd",
            "norm_q_action",
            "norm_q_torque",
        ]
        norm_present = [k for k in norm_keys if k in self.data]
        norm_labels = [
            "vx",
            "vy",
            "orient",
            "acc",
            "L_frc",
            "R_frc",
            "L_spd",
            "R_spd",
            "action",
            "torque",
        ]
        norm_labels = [
            norm_labels[i] for i, k in enumerate(norm_keys) if k in norm_present
        ]
        self.lines.update(
            create_lines(
                ax_norm,
                norm_present,
                sns.color_palette("tab10", len(norm_present)),
                labels=norm_labels,
            )
        )

        self._finalize_axes()
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.subplots_adjust(hspace=0.3, wspace=0.25)

    def update_data(self):
        action = self.env.action_space.sample()
        self.observation, reward, self.terminated, self.truncated, info = self.env.step(
            action
        )

        if self.truncated:
            self.env.reset()

        self._update_render()
        self._trim_all()
        self.data["timestep"].append(self.timestep)

        # Context
        self._store_context_data()
        pelvis_quat = self.env.sensor_data["framequat"]
        self._store_rpy(pelvis_quat)

        # Metrics from info
        if "other_metrics" in info:
            metrics = info["other_metrics"]
            for prefix, metrics_key in [
                ("raw_", "raw_quantities"),
                ("norm_", "normalized_quantities"),
                ("exp_", "after_exponential"),
            ]:
                if metrics_key in metrics:
                    for key, val in metrics[metrics_key].items():
                        data_key = f"{prefix}{key}"
                        if data_key in self.data:
                            self.data[data_key].append(val)

            if "rewards" in metrics:
                self.data["r_biped"].append(metrics["rewards"].get("r_biped", np.nan))
                self.data["r_cmd"].append(metrics["rewards"].get("r_cmd", np.nan))
                self.data["r_smooth"].append(metrics["rewards"].get("r_smooth", np.nan))

            self.data["total_reward"].append(reward)
        else:
            for key in self.data:
                if key.startswith(("raw_", "norm_", "exp_")):
                    self.data[key].append(np.nan)
            self.data["r_biped"].append(np.nan)
            self.data["r_cmd"].append(np.nan)
            self.data["r_smooth"].append(np.nan)
            self.data["total_reward"].append(np.nan)

        # Phase coefficients
        phi = self.env.phi
        spc = self.env.steps_per_cycle
        idx_l = int(round(((phi + THETA_LEFT) % 1.0) * spc)) % spc
        idx_r = int(round(((phi + THETA_RIGHT) % 1.0) * spc)) % spc

        self.data["c_frc_left"].append(
            c_swing_frc * self.env.von_mises_values_swing[idx_l]
        )
        self.data["c_frc_right"].append(
            c_swing_frc * self.env.von_mises_values_swing[idx_r]
        )
        self.data["c_spd_left"].append(
            c_stance_spd * self.env.von_mises_values_stance[idx_l]
        )
        self.data["c_spd_right"].append(
            c_stance_spd * self.env.von_mises_values_stance[idx_r]
        )

        self.timestep += 1


if __name__ == "__main__":
    monitor = RewardCassieMonitor(max_data_points=400)
    monitor.run(title="Live Cassie Reward Monitor", frames=100, interval=40)
