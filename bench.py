import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.env.cassie import CassieEnv
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import mujoco as mj  # Import mujoco
import cv2  # Import cv2 for resize if needed

# Create a high-quality plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (24, 12)  # Increased figure size for 3x4 layout
plt.rcParams["figure.dpi"] = 120  # Slightly lower DPI if needed for performance
# plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('viridis', 10)) # Commented out global cycler
plt.rcParams["lines.linewidth"] = 1.5  # Slightly thinner lines for clarity
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8  # Smaller font for more plots
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7


class LiveCassieMonitor:
    def __init__(self, max_data_points=200):  # Increased default points
        # Initialize environment
        self.env = CassieEnv(
            env_config={
                "symmetric_regulation": False,
                "sim_fps": 40,  # Match interval in run() -> 1000/40 = 25 fps simulation rate
            }
        )
        self.max_data_points = max_data_points

        # Get body IDs from model
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

        # Define observation structure (Indices based on cassie.py _set_obs concatenation order)
        # Total size: 39
        self.obs_slices = {
            "actuatorpos": slice(0, 10),  # 10
            "jointpos": slice(10, 16),  # 6
            "framequat": slice(16, 20),  # 4
            "gyro": slice(20, 23),  # 3
            "accelerometer": slice(23, 26),  # 3
            "magnetometer": slice(26, 29),  # 3
            "command": slice(29, 31),  # 2
            "contact_forces": slice(31, 37),  # 6 (Left XYZ, Right XYZ)
            "clock": slice(37, 39),  # 2
        }

        # Define names corresponding to slices (ensure order matches observation vector)
        self.obs_component_names = {
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
            "contact_forces": [
                "left_x",
                "left_y",
                "left_z",
                "right_x",
                "right_y",
                "right_z",
            ],
            "clock": ["sin", "cos"],
        }

        # Initialize data structure dynamically
        self.data = {"timestep": []}
        for category, names in self.obs_component_names.items():
            for name in names:
                self.data[f"{category}_{name}"] = []

        # Add non-observation data and rewards
        self.data.update(
            {
                "pelvis_height": [],
                "left_foot_height": [],
                "right_foot_height": [],
                "pelvis_roll": [],
                "pelvis_pitch": [],
                "pelvis_yaw": [],
                "feet_distance_x": [],
                "feet_distance_y": [],
                # 'feet_distance_z': [] # Z distance often less informative than X/Y for walking
                "total_reward": [],
                "r_biped": [],
                "r_cmd": [],
                "r_smooth": [],
            }
        )

        # Setup the figure and axes for live plots
        self.setup_figure()

        # Reset environment and initialize state
        self.observation, _ = self.env.reset()
        self.timestep = 0
        self.last_render = np.zeros(
            (self.env.height, self.env.width, 3), dtype=np.uint8
        )  # Init render frame
        self.terminated = False
        self.truncated = False

    def setup_figure(self):
        # Create the main figure with 3 rows, 4 columns layout
        self.fig = plt.figure(figsize=plt.rcParams["figure.figsize"])
        # Give video slightly more relative width
        self.gs = GridSpec(
            3, 4, figure=self.fig, width_ratios=[1.5, 1, 1, 1], height_ratios=[1, 1, 1]
        )

        # --- Column 0: Video ---
        self.ax_video = self.fig.add_subplot(self.gs[:, 0])
        self.ax_video.set_title("Cassie Simulation")
        self.ax_video.axis("off")
        # Initialize with correct aspect ratio if possible
        render_height, render_width = self.env.height, self.env.width
        self.img_display = self.ax_video.imshow(
            np.zeros((render_height, render_width, 3), dtype=np.uint8)
        )

        # --- Data Plots (3x3 grid in columns 1-3) ---

        # Row 0
        self.ax_motors_left = self.fig.add_subplot(self.gs[0, 1])
        self.ax_motors_left.set_title("Left Leg Motors")
        self.ax_motors_left.set_ylabel("Position (deg)")
        self.ax_motors_left.tick_params(axis="x", labelbottom=False)

        self.ax_imu = self.fig.add_subplot(self.gs[0, 2])
        self.ax_imu.set_title("Pelvis Orientation (RPY)")
        self.ax_imu.set_ylabel("Angle (deg)")
        self.ax_imu.tick_params(axis="x", labelbottom=False)

        self.ax_heights = self.fig.add_subplot(self.gs[0, 3])
        self.ax_heights.set_title("Heights")
        self.ax_heights.set_ylabel("Height (m)")
        self.ax_heights.tick_params(axis="x", labelbottom=False)

        # Row 1
        self.ax_motors_right = self.fig.add_subplot(self.gs[1, 1])
        self.ax_motors_right.set_title("Right Leg Motors")
        self.ax_motors_right.set_ylabel("Position (deg)")
        self.ax_motors_right.tick_params(axis="x", labelbottom=False)

        self.ax_gyro = self.fig.add_subplot(self.gs[1, 2])  # New: Gyro
        self.ax_gyro.set_title("Gyroscope")
        self.ax_gyro.set_ylabel("Ang. Vel (rad/s)")
        self.ax_gyro.tick_params(axis="x", labelbottom=False)

        self.ax_forces = self.fig.add_subplot(self.gs[1, 3])
        self.ax_forces.set_title("Contact Forces (World Frame)")
        self.ax_forces.set_ylabel("Force (N)")
        self.ax_forces.tick_params(axis="x", labelbottom=False)

        # Row 2
        self.ax_jointpos = self.fig.add_subplot(self.gs[2, 1])  # New: Joint Positions
        self.ax_jointpos.set_title("Joint Positions")
        self.ax_jointpos.set_ylabel("Position (deg)")
        self.ax_jointpos.set_xlabel("Timestep")

        self.ax_accel = self.fig.add_subplot(self.gs[2, 2])  # New: Accelerometer
        self.ax_accel.set_title("Accelerometer")
        self.ax_accel.set_ylabel("Lin. Accel (m/s^2)")
        self.ax_accel.set_xlabel("Timestep")

        self.ax_rewards = self.fig.add_subplot(
            self.gs[2, 3]
        )  # New: Rewards plot replacing Misc
        self.ax_rewards.set_title("Rewards")
        self.ax_rewards.set_ylabel("Reward Value")
        self.ax_rewards.set_xlabel("Timestep")

        # Create line objects dynamically
        self.lines = {}
        self.all_axes = [  # Keep track of all data axes
            self.ax_motors_left,
            self.ax_imu,
            self.ax_heights,
            self.ax_motors_right,
            self.ax_gyro,
            self.ax_forces,
            self.ax_jointpos,
            self.ax_accel,
            self.ax_rewards,  # Added rewards axis
        ]

        # Left Motors (gs[0, 1], Palette: tab10)
        left_motor_keys = [
            f"actuatorpos_{name}"
            for name in self.obs_component_names["actuatorpos"][:5]
        ]
        self._create_lines(
            self.ax_motors_left,
            left_motor_keys,
            sns.color_palette("tab10", 5),
            key_suffix=True,
        )

        # IMU RPY (gs[0, 2], Palette: Accent)
        imu_keys = ["pelvis_roll", "pelvis_pitch", "pelvis_yaw"]
        self._create_lines(self.ax_imu, imu_keys, sns.color_palette("Accent", 3))

        # Heights (gs[0, 3], Palette: Paired)
        height_keys = ["pelvis_height", "left_foot_height", "right_foot_height"]
        self._create_lines(self.ax_heights, height_keys, sns.color_palette("Paired", 3))

        # Right Motors (gs[1, 1], Palette: Set2)
        right_motor_keys = [
            f"actuatorpos_{name}"
            for name in self.obs_component_names["actuatorpos"][5:]
        ]
        self._create_lines(
            self.ax_motors_right,
            right_motor_keys,
            sns.color_palette("Set2", 5),
            key_suffix=True,
        )

        # Gyroscope (gs[1, 2], Palette: Dark2)
        gyro_keys = [f"gyro_{name}" for name in self.obs_component_names["gyro"]]
        self._create_lines(
            self.ax_gyro, gyro_keys, sns.color_palette("Dark2", 3), key_suffix=True
        )

        # Contact Forces (gs[1, 3], Palette: tab10)
        force_keys = [
            f"contact_forces_{name}"
            for name in self.obs_component_names["contact_forces"]
        ]
        force_labels = ["L_Fx", "L_Fy", "L_Fz", "R_Fx", "R_Fy", "R_Fz"]
        self._create_lines(
            self.ax_forces,
            force_keys,
            sns.color_palette("tab10", 6),
            labels=force_labels,
        )

        # Joint Positions (gs[2, 1], Palette: Paired)
        jointpos_keys = [
            f"jointpos_{name}" for name in self.obs_component_names["jointpos"]
        ]
        self._create_lines(
            self.ax_jointpos,
            jointpos_keys,
            sns.color_palette("Paired", 6),
            key_suffix=True,
        )

        # Accelerometer (gs[2, 2], Palette: Set1)
        accel_keys = [
            f"accelerometer_{name}"
            for name in self.obs_component_names["accelerometer"]
        ]
        self._create_lines(
            self.ax_accel, accel_keys, sns.color_palette("Set1", 3), key_suffix=True
        )

        # Misc -> Rewards (gs[2, 3], Palette: Dark2)
        reward_keys = ["total_reward", "r_biped", "r_cmd", "r_smooth"]
        reward_labels = ["Total", "Biped", "Cmd", "Smooth"]
        self._create_lines(
            self.ax_rewards,
            reward_keys,
            sns.color_palette("Dark2", 4),
            labels=reward_labels,
        )

        # Set common x-limits and grid for all data plots initially
        for ax in self.all_axes:
            ax.set_xlim(0, self.max_data_points)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right")  # Add legend to all

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout rect
        plt.subplots_adjust(hspace=0.2, wspace=0.25)  # Adjust spacing

    def _create_lines(self, ax, keys, colors, labels=None, key_suffix=False):
        """Helper function to create line objects for an axis."""
        if labels is None:
            labels = keys
        elif len(labels) != len(keys):
            raise ValueError("Number of labels must match number of keys")

        for i, key in enumerate(keys):
            label = (
                labels[i].split("_")[-1]
                if key_suffix and "_range" not in labels[i]
                else labels[i]
            )
            (self.lines[key],) = ax.plot([], [], label=label, color=colors[i])

    def update_data(self):
        # Use a simple standing action or random walk for testing
        # action = np.zeros(self.env.action_space.shape[0]) # Standing
        action = np.random.uniform(
            -0.1, 0.1, self.env.action_space.shape[0]
        )  # Random walk

        # # Check for termination/truncation before stepping
        # if self.terminated or self.truncated:
        #      print("Episode finished. Resetting.")
        #      self.observation, _ = self.env.reset()
        #      self.timestep = 0
        #      # Clear data
        #      for key in self.data:
        #          self.data[key] = []
        #      self.terminated = False
        #      self.truncated = False

        self.observation, reward, self.terminated, self.truncated, info = self.env.step(
            action
        )

        # Get render frame
        render_result = self.env.render()
        if render_result is not None and isinstance(render_result, np.ndarray):
            # Ensure frame dimensions match imshow extent
            if render_result.shape[0:2] == (self.env.height, self.env.width):
                self.last_render = render_result
            else:
                # Resize if necessary (though render() should provide correct size)
                self.last_render = cv2.resize(
                    render_result, (self.env.width, self.env.height)
                )

        # Store timestep and trim data
        current_length = len(self.data["timestep"])
        if current_length >= self.max_data_points:
            # Trim all data lists efficiently
            for key in self.data:
                self.data[key].pop(0)  # Remove oldest point
        self.data["timestep"].append(self.timestep)

        # Store position data using body IDs
        self.data["pelvis_height"].append(self.env.data.xpos[self.pelvis_id, 2])
        self.data["left_foot_height"].append(self.env.data.xpos[self.left_foot_id, 2])
        self.data["right_foot_height"].append(self.env.data.xpos[self.right_foot_id, 2])

        # Store distance metrics
        left_foot_pos = self.env.data.xpos[self.left_foot_id]
        right_foot_pos = self.env.data.xpos[self.right_foot_id]
        self.data["feet_distance_x"].append(abs(left_foot_pos[0] - right_foot_pos[0]))
        self.data["feet_distance_y"].append(abs(left_foot_pos[1] - right_foot_pos[1]))
        # self.data['feet_distance_z'].append(abs(left_foot_pos[2] - right_foot_pos[2]))

        # Parse and store observation data using slices
        for category, obs_slice in self.obs_slices.items():
            names = self.obs_component_names[category]
            values = self.observation[obs_slice]
            for i, name in enumerate(names):
                # Handle scalar vs vector slices
                val = (
                    values[i]
                    if isinstance(values, np.ndarray) and values.ndim > 0
                    else values
                )
                self.data[f"{category}_{name}"].append(val)

        # Extract orientation and convert to RPY (degrees)
        pelvis_quaternion = self.observation[self.obs_slices["framequat"]]
        # Ensure the quaternion is normalized before conversion
        norm = np.linalg.norm(pelvis_quaternion)
        if norm > 1e-6:
            pelvis_quaternion /= norm
        else:
            # Handle zero quaternion case if necessary, e.g., set to identity
            pelvis_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        pelvis_rpy = self.env.quat_to_rpy(
            pelvis_quaternion, radians=False
        )  # Use env's method

        self.data["pelvis_roll"].append(pelvis_rpy[0])
        self.data["pelvis_pitch"].append(pelvis_rpy[1])
        self.data["pelvis_yaw"].append(pelvis_rpy[2])

        # Append reward data
        # Ensure 'custom_rewards' exists and has the expected keys
        if "custom_rewards" in info and all(
            k in info["custom_rewards"] for k in ["r_biped", "r_cmd", "r_smooth"]
        ):
            self.data["total_reward"].append(reward)
            self.data["r_biped"].append(info["custom_rewards"]["r_biped"])
            self.data["r_cmd"].append(info["custom_rewards"]["r_cmd"])
            self.data["r_smooth"].append(info["custom_rewards"]["r_smooth"])
        else:
            # Append default values (e.g., NaN or 0) if rewards are missing
            log_msg = "Reward data missing or incomplete in info dict. Appending NaN."
            # Avoid printing too frequently if this happens often
            if self.timestep % 50 == 0:  # Log every 50 steps
                print(log_msg)
            self.data["total_reward"].append(np.nan)
            self.data["r_biped"].append(np.nan)
            self.data["r_cmd"].append(np.nan)
            self.data["r_smooth"].append(np.nan)

        self.timestep += 1

    def update_plot(self, frame):
        # Update data from simulation
        try:
            self.update_data()
        except Exception as e:
            print(f"Error during data update: {e}")
            # Optionally stop animation or handle error
            # self.ani.event_source.stop()
            return []

        artists = []

        # Update the video frame
        self.img_display.set_array(self.last_render)
        artists.append(self.img_display)

        # Update all plot lines
        timesteps = self.data["timestep"]

        if not timesteps:  # Skip update if no data yet
            return artists

        for key, line in self.lines.items():
            if key in self.data:
                line.set_data(timesteps, self.data[key])
                artists.append(line)

        # Update plot limits dynamically based on current view
        min_time = timesteps[0]
        max_time = timesteps[-1]
        if max_time == min_time:  # Avoid zero range
            max_time += 1

        # Update all data axes
        for ax in self.all_axes:
            ax.set_xlim(min_time, max_time)
            # Auto-scale Y axis
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            # No need to extend artists here, already done in the loop over self.lines

        return artists

    def run(self, frames=1000, interval=40):  # interval matches sim_fps reciprocal
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=frames,  # Set frames to None for indefinite run? Or handle termination
            interval=interval,  # Update interval in ms
            blit=True,
            repeat=False,  # Don't repeat animation automatically
        )
        self.fig.suptitle("Live Cassie Robot Monitor", fontsize=14)
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")
        finally:
            # Ensure environment is closed if plot window is closed
            print("Closing environment.")
            self.env.close()


# Create and run the live monitor
if __name__ == "__main__":
    monitor = LiveCassieMonitor(max_data_points=300)  # Slightly more history
    # Run indefinitely until window is closed or max_steps reached in env
    monitor.run(frames=50, interval=40)
