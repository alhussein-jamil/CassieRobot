import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.env.cassie import CassieEnv
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import mujoco as mj
import cv2
from src.env.constants import (
    THETA_LEFT,
    THETA_RIGHT,
    c_stance_spd,
    c_swing_frc,
)

# Create a high-quality plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (24, 12)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7


class RewardCassieMonitor:
    def __init__(self, max_data_points=300):
        # Initialize environment
        self.env = CassieEnv(
            env_config={
                "symmetric_regulation": False,
                "sim_fps": 40,
            }
        )
        self.max_data_points = max_data_points

        # Get body IDs from model (still useful for context plots like height)
        try:
            self.pelvis_id = mj.mj_name2id(
                self.env.model, mj.mjtObj.mjOBJ_BODY, "cassie-pelvis"
            )
            self.left_foot_id = mj.mj_name2id(
                self.env.model, mj.mjtObj.mjOBJ_BODY, "left-foot"
            )
            self.right_foot_id = mj.mj_name2id(
                self.env.model, mj.mjtObj.mjOBJ_BODY, "right-foot"
            )
        except ValueError as e:
            print(
                f"Error getting body IDs: {e}. Ensure the MuJoCo model contains 'cassie-pelvis', 'left-foot', 'right-foot'."
            )
            raise

        # Initialize data structure for reward components and related values
        self.data = {
            "timestep": [],
            "pelvis_height": [],  # Keep for context
            "left_foot_height": [],  # Keep for context
            "right_foot_height": [],  # Keep for context
            "pelvis_roll": [],  # Keep for context
            "pelvis_pitch": [],  # Keep for context
            "pelvis_yaw": [],  # Keep for context
            # Raw Quantities
            "raw_q_vx": [],
            "raw_q_vy": [],
            "raw_q_left_frc": [],
            "raw_q_right_frc": [],
            "raw_q_left_spd": [],
            "raw_q_right_spd": [],
            "raw_q_action": [],
            "raw_q_pelvis_acc": [],
            "raw_q_torque": [],
            "raw_q_orientation": [],
            # Normalized Quantities
            "norm_q_vx": [],
            "norm_q_vy": [],
            "norm_q_left_frc": [],
            "norm_q_right_frc": [],
            "norm_q_left_spd": [],
            "norm_q_right_spd": [],
            "norm_q_action": [],
            "norm_q_pelvis_acc": [],
            "norm_q_torque": [],
            "norm_q_orientation": [],
            # Exponential Quantities (after exp(-omega*norm_q))
            # "exp_q_vx": [],
            # "exp_q_vy": [],
            # "exp_q_left_frc": [],
            # "exp_q_right_frc": [],
            # "exp_q_left_spd": [],
            # "exp_q_right_spd": [],
            # "exp_q_action": [],
            "exp_q_pelvis_acc": [],
            # "exp_q_torque": [],
            # "exp_q_orientation": [],
            # Phase Coefficients
            "c_frc_left": [],
            "c_frc_right": [],
            "c_spd_left": [],
            "c_spd_right": [],
            # Final Rewards
            "total_reward": [],
            "r_biped": [],
            "r_cmd": [],
            "r_smooth": [],
        }

        # Setup the figure and axes for live plots
        self.setup_figure()

        # Reset environment and initialize state
        self.observation, self.reset_info = (
            self.env.reset()
        )  # Store reset_info if needed later
        self.timestep = 0
        self.last_render = np.zeros(
            (self.env.height, self.env.width, 3), dtype=np.uint8
        )
        self.terminated = False
        self.truncated = False

    def setup_figure(self):
        self.fig = plt.figure(figsize=plt.rcParams["figure.figsize"])
        self.gs = GridSpec(
            4,
            4,
            figure=self.fig,
            width_ratios=[1.5, 1, 1, 1],
            height_ratios=[1, 1, 1, 1],
        )

        # --- Column 0: Video ---
        self.ax_video = self.fig.add_subplot(self.gs[0:3, 0])
        self.ax_video.set_title("Cassie Simulation")
        self.ax_video.axis("off")
        render_height, render_width = self.env.height, self.env.width
        self.img_display = self.ax_video.imshow(
            np.zeros((render_height, render_width, 3), dtype=np.uint8)
        )

        # --- Data Plots (3x3 grid in columns 1-3) ---

        # Row 0: Raw Quantities (Vel, Orient, Accel) & Heights for context
        self.ax_raw_vel = self.fig.add_subplot(self.gs[0, 1])
        self.ax_raw_vel.set_title("Raw Pelvis Vel Err (q_vx, q_vy)")
        self.ax_raw_vel.set_ylabel("Abs Vel Err (m/s)")
        self.ax_raw_vel.tick_params(axis="x", labelbottom=False)

        self.ax_raw_orient_accel = self.fig.add_subplot(self.gs[0, 2])
        self.ax_raw_orient_accel.set_title("Raw Orient Err & Pelvis Acc (q)")
        self.ax_raw_orient_accel.set_ylabel("Norm")
        self.ax_raw_orient_accel.tick_params(axis="x", labelbottom=False)

        self.ax_heights = self.fig.add_subplot(
            self.gs[0, 3]
        )  # Keep heights for context
        self.ax_heights.set_title("Heights & Pelvis RPY")
        self.ax_heights.set_ylabel("Meters / Deg")
        self.ax_heights.tick_params(axis="x", labelbottom=False)

        # Row 1: Raw Quantities (Forces, Speeds, Action) & Phase Coeffs
        self.ax_raw_forces = self.fig.add_subplot(self.gs[1, 1])
        self.ax_raw_forces.set_title("Raw Foot Forces (q_frc)")
        self.ax_raw_forces.set_ylabel("Force Norm (N)")
        self.ax_raw_forces.tick_params(axis="x", labelbottom=False)

        self.ax_raw_speeds = self.fig.add_subplot(self.gs[1, 2])
        self.ax_raw_speeds.set_title("Raw Foot Speeds (q_spd)")
        self.ax_raw_speeds.set_ylabel("Abs Speed (rad/s?)")  # Check unit
        self.ax_raw_speeds.tick_params(axis="x", labelbottom=False)

        self.ax_phase_coeffs = self.fig.add_subplot(self.gs[1, 3])
        self.ax_phase_coeffs.set_title("Phase Coefficients (c_frc, c_spd)")
        self.ax_phase_coeffs.set_ylabel("Coefficient Value")
        self.ax_phase_coeffs.tick_params(axis="x", labelbottom=False)

        # Row 2: Exponential Quantities & Rewards
        self.ax_exp_quantities = self.fig.add_subplot(self.gs[2, 1])
        self.ax_exp_quantities.set_title("Exponential Quantities (exp(-omega*q))")
        self.ax_exp_quantities.set_ylabel("Value (0-1)")
        self.ax_exp_quantities.set_xlabel("Timestep")

        self.ax_rewards_comp = self.fig.add_subplot(self.gs[2, 2])
        self.ax_rewards_comp.set_title("Reward Components")
        self.ax_rewards_comp.set_ylabel("Reward Value")
        self.ax_rewards_comp.set_xlabel("Timestep")

        self.ax_total_reward = self.fig.add_subplot(self.gs[2, 3])
        self.ax_total_reward.set_title("Total Reward")
        self.ax_total_reward.set_ylabel("Reward Value")
        self.ax_total_reward.tick_params(axis="x", labelbottom=False)

        # Row 3: All Normalized Quantities (Combined)
        self.ax_norm_all = self.fig.add_subplot(
            self.gs[3, :]
        )  # Span all columns in row 3
        self.ax_norm_all.set_title("Normalized Quantities (0-1)")
        self.ax_norm_all.set_ylabel("Norm Value")
        self.ax_norm_all.set_xlabel("Timestep")

        # Create line objects dynamically
        self.lines = {}
        self.all_axes = [
            self.ax_raw_vel,
            self.ax_raw_orient_accel,
            self.ax_heights,
            self.ax_raw_forces,
            self.ax_raw_speeds,
            self.ax_phase_coeffs,
            self.ax_exp_quantities,
            self.ax_rewards_comp,
            self.ax_total_reward,
            self.ax_norm_all,
        ]

        # --- Create lines for each plot ---

        # Raw Vel (gs[0, 1])
        raw_vel_keys = ["raw_q_vx", "raw_q_vy"]
        self._create_lines(
            self.ax_raw_vel,
            raw_vel_keys,
            sns.color_palette("viridis", 2),
            labels=["q_vx", "q_vy"],
        )

        # Raw Orient & Accel (gs[0, 2])
        raw_orient_accel_keys = ["raw_q_orientation", "raw_q_pelvis_acc"]
        self._create_lines(
            self.ax_raw_orient_accel,
            raw_orient_accel_keys,
            sns.color_palette("plasma", 2),
            labels=["q_orient", "q_pelvis_acc"],
        )

        # Heights & RPY (gs[0, 3]) - Context
        height_keys = [
            "pelvis_height",
            "left_foot_height",
            "right_foot_height",
            "pelvis_roll",
            "pelvis_pitch",
            "pelvis_yaw",
        ]
        height_labels = ["Pelvis", "L Foot", "R Foot", "Roll", "Pitch", "Yaw"]
        self._create_lines(
            self.ax_heights,
            height_keys,
            sns.color_palette("Paired", 6),
            labels=height_labels,
        )

        # Raw Forces (gs[1, 1])
        raw_force_keys = ["raw_q_left_frc", "raw_q_right_frc"]
        self._create_lines(
            self.ax_raw_forces,
            raw_force_keys,
            sns.color_palette("coolwarm", 2),
            labels=["q_L_frc", "q_R_frc"],
        )

        # Raw Speeds (gs[1, 2])
        raw_speed_keys = ["raw_q_left_spd", "raw_q_right_spd"]
        self._create_lines(
            self.ax_raw_speeds,
            raw_speed_keys,
            sns.color_palette("PRGn", 2),
            labels=["q_L_spd", "q_R_spd"],
        )

        # Phase Coeffs (gs[1, 3])
        phase_coeff_keys = ["c_frc_left", "c_frc_right", "c_spd_left", "c_spd_right"]
        phase_labels = ["c_frc_L", "c_frc_R", "c_spd_L", "c_spd_R"]
        self._create_lines(
            self.ax_phase_coeffs,
            phase_coeff_keys,
            sns.color_palette("Spectral", 4),
            labels=phase_labels,
        )

        # Exp Quantities (gs[2, 1]) - Plot a selection
        exp_keys_plot = [
            "exp_q_vx",
            "exp_q_vy",
            "exp_q_left_frc",
            "exp_q_right_frc",
            "exp_q_action",
            "exp_q_orientation",
            "exp_q_pelvis_acc",
            "exp_q_torque",
        ]
        exp_labels = [
            "exp_vx",
            "exp_vy",
            "exp_L_frc",
            "exp_R_frc",
            "exp_action",
            "exp_orient",
            "exp_pelvis_acc",
            "exp_torque",
        ]
        self._create_lines(
            self.ax_exp_quantities,
            exp_keys_plot,
            sns.color_palette("tab10", 8),
            labels=exp_labels,
        )

        # Reward Components (gs[2, 2])
        reward_comp_keys = ["r_biped", "r_cmd", "r_smooth"]
        self._create_lines(
            self.ax_rewards_comp,
            reward_comp_keys,
            sns.color_palette("Accent", 3),
            labels=["Biped", "Cmd", "Smooth"],
        )

        # Total Reward (gs[2, 3])
        self._create_lines(
            self.ax_total_reward,
            ["total_reward"],
            sns.color_palette("Set1", 1),
            labels=["Total"],
        )

        # All Normalized Quantities (gs[3, :]) - Combined Plot
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
            "norm_q_torque",  # Added action and torque if available
        ]
        # Filter keys that actually exist in self.data to avoid errors if some aren't tracked
        norm_keys_present = [k for k in norm_keys if k in self.data]
        norm_labels = [
            "vx",
            "vy",
            "orient",
            "pelvis_acc",
            "L_frc",
            "R_frc",
            "L_spd",
            "R_spd",
            "action",
            "torque",
        ]
        # Filter labels corresponding to present keys
        norm_labels_present = [
            norm_labels[i] for i, k in enumerate(norm_keys) if k in norm_keys_present
        ]

        # Use a larger, distinct color palette
        num_norm_lines = len(norm_keys_present)
        norm_colors = sns.color_palette(
            "tab10", num_norm_lines
        )  # Or "husl", "viridis", etc.

        self._create_lines(
            self.ax_norm_all, norm_keys_present, norm_colors, labels=norm_labels_present
        )

        # Set common x-limits and grid for all data plots initially
        for ax in self.all_axes:
            ax.set_xlim(0, self.max_data_points)
            ax.grid(True, linestyle="--", alpha=0.6)
            if ax.get_lines():  # Only add legend if there are lines
                ax.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.subplots_adjust(hspace=0.3, wspace=0.25)

    def _create_lines(self, ax, keys, colors, labels=None, key_suffix=False):
        """Helper function to create line objects for an axis."""
        if labels is None:
            # Auto-generate labels from keys if not provided
            labels = [
                k.replace("raw_", "")
                .replace("norm_", "")
                .replace("exp_", "")
                .replace("_", " ")
                .title()
                for k in keys
            ]
        elif len(labels) != len(keys):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of keys ({len(keys)}) for axis {ax.get_title()}"
            )

        for i, key in enumerate(keys):
            label_text = labels[i]
            (self.lines[key],) = ax.plot([], [], label=label_text, color=colors[i])

    def update_data(self):
        # Use a simple standing action or random walk for testing
        action = np.random.uniform(-0.1, 0.1, self.env.action_space.shape[0]).astype(
            np.float32
        )
        action = self.env.action_space.sample()

        # Step the environment
        self.observation, reward, self.terminated, self.truncated, info = self.env.step(
            action
        )

        if self.truncated:
            self.env.reset()

        # Get render frame
        render_result = self.env.render()
        if render_result is not None and isinstance(render_result, np.ndarray):
            if render_result.shape[0:2] == (self.env.height, self.env.width):
                self.last_render = render_result
            else:
                self.last_render = cv2.resize(
                    render_result, (self.env.width, self.env.height)
                )

        # Store timestep and trim data
        current_length = len(self.data["timestep"])
        if current_length >= self.max_data_points:
            for key in self.data:
                self.data[key].pop(0)
        self.data["timestep"].append(self.timestep)

        # --- Store Context Data (Heights, RPY) ---
        self.data["pelvis_height"].append(self.env.data.xpos[self.pelvis_id, 2])
        self.data["left_foot_height"].append(self.env.data.xpos[self.left_foot_id, 2])
        self.data["right_foot_height"].append(self.env.data.xpos[self.right_foot_id, 2])

        # Get current pelvis orientation RPY
        pelvis_quaternion = self.env.sensor_data["framequat"]
        pelvis_rpy = self.env.quat_to_rpy(
            pelvis_quaternion, radians=False
        )  # Degrees for plotting
        self.data["pelvis_roll"].append(pelvis_rpy[0])
        self.data["pelvis_pitch"].append(pelvis_rpy[1])
        self.data["pelvis_yaw"].append(pelvis_rpy[2])

        # --- Store Reward-Related Data from info['other_metrics'] ---
        if "other_metrics" in info:
            metrics = info["other_metrics"]

            # Raw Quantities
            if "raw_quantities" in metrics:
                for key, val in metrics["raw_quantities"].items():
                    data_key = f"raw_{key}"
                    if data_key in self.data:
                        self.data[data_key].append(val)  # Append directly
                    else:
                        # Only print warning once per missing key
                        if not hasattr(self, f"_warned_{data_key}"):
                            print(
                                f"Warning: Metric key '{data_key}' not found in self.data"
                            )
                            setattr(self, f"_warned_{data_key}", True)

            # Normalized Quantities (Optional to plot, but store if needed)
            if "normalized_quantities" in metrics:
                for key, val in metrics["normalized_quantities"].items():
                    data_key = f"norm_{key}"
                    if data_key in self.data:
                        self.data[data_key].append(val)  # Append directly

            # Exponential Quantities
            if "after_exponential" in metrics:
                for key, val in metrics["after_exponential"].items():
                    data_key = f"exp_{key}"
                    if data_key in self.data:
                        self.data[data_key].append(val)  # Append directly

            # Reward Components
            if "rewards" in metrics:
                self.data["r_biped"].append(metrics["rewards"].get("r_biped", np.nan))
                self.data["r_cmd"].append(metrics["rewards"].get("r_cmd", np.nan))
                self.data["r_smooth"].append(metrics["rewards"].get("r_smooth", np.nan))

            # Total Reward (from step return)
            self.data["total_reward"].append(reward)

        else:
            # Append NaNs if metrics are missing
            print(
                f"Warning: 'other_metrics' not found in info dict at timestep {self.timestep}"
            )
            for key_prefix in ["raw_", "norm_", "exp_"]:
                for data_key in self.data:
                    if data_key.startswith(key_prefix):
                        self.data[data_key].append(np.nan)
            self.data["r_biped"].append(np.nan)
            self.data["r_cmd"].append(np.nan)
            self.data["r_smooth"].append(np.nan)
            self.data["total_reward"].append(np.nan)

        # Calculate and store phase coefficients
        phi = self.env.phi
        steps_per_cycle = self.env.steps_per_cycle
        # Use integer indexing, ensuring safety with modulo
        idx_left = (
            int(round(((phi + THETA_LEFT) % 1.0) * steps_per_cycle)) % steps_per_cycle
        )
        idx_right = (
            int(round(((phi + THETA_RIGHT) % 1.0) * steps_per_cycle)) % steps_per_cycle
        )

        # Access precomputed Von Mises values from the environment instance
        i_swing_left = self.env.von_mises_values_swing[idx_left]
        i_swing_right = self.env.von_mises_values_swing[idx_right]
        i_stance_left = self.env.von_mises_values_stance[idx_left]
        i_stance_right = self.env.von_mises_values_stance[idx_right]

        self.data["c_frc_left"].append(c_swing_frc * i_swing_left)
        self.data["c_frc_right"].append(c_swing_frc * i_swing_right)
        self.data["c_spd_left"].append(c_stance_spd * i_stance_left)
        self.data["c_spd_right"].append(c_stance_spd * i_stance_right)

        self.timestep += 1

    def update_plot(self, frame):
        # Update data from simulation
        try:
            self.update_data()
        except Exception as e:
            print(f"Error during data update: {e}")
            # self.ani.event_source.stop() # Optional: stop on error
            return []  # Return empty list of artists on error

        artists = []

        # Update the video frame
        self.img_display.set_array(self.last_render)
        artists.append(self.img_display)

        # Update all plot lines
        timesteps = self.data["timestep"]

        if not timesteps:
            return artists  # Skip update if no data

        min_time = timesteps[0]
        max_time = timesteps[-1]
        if max_time == min_time:
            max_time += 1  # Avoid zero range if only one point

        # Update lines using the stored Line2D objects
        for key, line in self.lines.items():
            if key in self.data and len(self.data[key]) == len(timesteps):
                line.set_data(timesteps, self.data[key])
                artists.append(line)
            elif key in self.data:
                # This might happen if a metric wasn't available initially
                # Pad with NaN or handle appropriately if lengths mismatch
                print(
                    f"Warning: Length mismatch for key '{key}'. Timesteps: {len(timesteps)}, Data: {len(self.data[key])}"
                )
                # Simple padding:
                padded_data = (
                    [np.nan] * (len(timesteps) - len(self.data[key]))
                ) + self.data[key]
                line.set_data(timesteps, padded_data)
                artists.append(line)

        # Update plot limits dynamically
        for ax in self.all_axes:
            ax.set_xlim(min_time, max_time)
            ax.relim()  # Recalculate data limits
            ax.autoscale_view(scalex=False, scaley=True)  # Autoscale Y only

        return artists

    def run(self, frames=1000, interval=40):
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=frames,
            interval=interval,
            blit=True,  # Use blitting for performance
            repeat=False,
        )
        self.fig.suptitle("Live Cassie Reward Monitor", fontsize=14)
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")
        finally:
            print("Closing environment.")
            self.env.close()


# Create and run the reward monitor
if __name__ == "__main__":
    monitor = RewardCassieMonitor(max_data_points=400)  # More history points
    monitor.run(
        frames=100, interval=40
    )  # Run indefinitely (or until env terminates/truncates)
