import logging as log
from copy import deepcopy
from typing import Any, TYPE_CHECKING, Dict, Tuple

import cv2
import mujoco as m
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from collections import OrderedDict
from pathlib import Path
from .constants import (
    DEFAULT_CONFIG,
    PELVIS,
    LEFT_FOOT,
    RIGHT_FOOT,
    LEFT_CONTACT_IDX,
    RIGHT_CONTACT_IDX,
    FORWARD_QUARTERNIONS,
    THETA_LEFT,
    THETA_RIGHT,
    c_stance_spd,
    c_swing_frc,
    sensors,
    mass,
    gravity,
    OMEGA,
)
from .functions import action_dist, p_between_von_mises, mod
import numba as nb

if TYPE_CHECKING:
    import numpy.typing as npt

log.basicConfig(level=log.INFO)


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 0,  # Will be calculated in __init__
    }
    mujoco_dt = 0.0005

    def __init__(
        self,
        env_config: Dict[str, Any] | None = None,
        model_dir: str | Path = "assets/model",
    ):
        # Combine user config with defaults
        config = deepcopy(DEFAULT_CONFIG)
        if env_config:
            config.update(env_config)

        # --- Configuration Parameters ---
        self.symmetric_regulation: bool = config["symmetric_regulation"]
        self._terminate_when_unhealthy: bool = config["terminate_when_unhealthy"]
        self._healthy_pelvis_z_range: Tuple[float, float] = config["pelvis_height"]
        self._healthy_feet_distance_x: float = config["feet_distance_x"]
        self._healthy_feet_distance_y: float = config["feet_distance_y"]
        self._healthy_feet_distance_z: float = config["feet_distance_z"]
        self._healthy_dis_to_pelvis: float = config["feet_pelvis_height"]
        self._healthy_feet_height: float = config["feet_height"]
        self._max_sim_time: int = config["max_simulation_time"]
        self.dt_per_cycle: float = config["dt_per_cycle"]
        self.max_roll: float = config["max_roll"]
        self.max_pitch: float = config["max_pitch"]
        self.max_yaw: float = config["max_yaw"]
        self.training: bool = config["is_training"]
        self.r: float = config["r"]  # Phase shift for stance phase distribution
        self.kappa: float = config[
            "kappa"
        ]  # Concentration parameter for Von Mises distributions
        self.x_cmd_vel: float = config["x_cmd_vel"]
        self.y_cmd_vel: float = config["y_cmd_vel"]
        self.model_file: str = config["model"]
        self._reset_noise_scale: float = config["reset_noise_scale"]
        self._force_max_norm: float = config["force_max_norm"]
        self._push_prob: float = config["push_prob"]
        self.render_width: int = config["width"]
        self.render_height: int = config["height"]
        self.sim_fps: int = config["sim_fps"]
        self.local_render_mode: str = config["render_mode"]

        # Extract reward coefficients (keys starting with 'r_' or 'bias')
        self.reward_coeffs = {
            k: v
            for k, v in config.items()
            if k.startswith("r_") or k.startswith("bias")
        }

        # --- Derived Parameters & State ---
        self.a_swing: float = 0.0  # Mean phase for swing force distribution
        self.a_stance: float = self.r  # Mean phase for stance speed distribution
        self.b_swing: float = self.a_stance  # End phase for swing force distribution
        self.b_stance: float = 1.0  # End phase for stance speed distribution
        self._pushing: int = (
            -1
        )  # Tracks duration of applied push force (-1: not pushing)

        self.phi: float = 0.0  # Current phase in the gait cycle [0, 1)
        self.steps: int = 0  # Number of simulation steps taken
        self.previous_action: npt.NDArray[np.float32] = np.zeros(
            10
        )  # Store previous action for smoothing reward
        self.command: npt.NDArray[np.float32] = np.array(
            [self.x_cmd_vel, self.y_cmd_vel]
        )  # Target velocity command
        self.contact: bool = False  # Flag indicating if feet have made contact recently
        self.symmetric_turn: bool = (
            False  # Flag to alternate between normal and symmetric steps
        )
        self.init_rpy: npt.NDArray[np.float32] | None = (
            None  # Initial roll, pitch, yaw of the pelvis
        )
        self.contact_force_left_foot: npt.NDArray[np.float32] = np.zeros(
            6
        )  # Contact force on left foot
        self.contact_force_right_foot: npt.NDArray[np.float32] = np.zeros(
            6
        )  # Contact force on right foot
        self.obs: npt.NDArray[np.float32] | None = None  # Current observation

        # --- Observation Space Definition ---
        observation_space_dict = OrderedDict(
            [
                ("actuatorpos", Box(-180.0, 180.0, shape=(10,))),  # Motor positions
                (
                    "jointpos",
                    Box(-180.0, 180.0, shape=(6,)),
                ),  # Joint positions (unactuated)
                (
                    "framequat",
                    Box(-1.0, 1.0, shape=(4,)),
                ),  # Pelvis orientation (quaternion)
                ("gyro", Box(-np.inf, np.inf, shape=(3,))),  # Pelvis angular velocity
                (
                    "accelerometer",
                    Box(-np.inf, np.inf, shape=(3,)),
                ),  # Pelvis linear acceleration
                (
                    "magnetometer",
                    Box(-np.inf, np.inf, shape=(3,)),
                ),  # Pelvis magnetic field vector
                ("command", Box(-np.inf, np.inf, shape=(2,))),  # Target x, y velocity
                (
                    "contact_forces",
                    Box(-np.inf, np.inf, shape=(6,)),
                ),  # Left and right foot contact forces (first 3 components each)
                (
                    "clock",
                    Box(-1.0, 1.0, shape=(2,)),
                ),  # Phase (sin(2*pi*phi), cos(2*pi*phi))
            ]
        )
        _obs_low = np.concatenate(
            [space.low for space in observation_space_dict.values()]
        )
        _obs_high = np.concatenate(
            [space.high for space in observation_space_dict.values()]
        )
        observation_space = Box(low=_obs_low, high=_obs_high, dtype=np.float32)

        # --- MujocoEnv Initialization ---
        frame_skip = int((1.0 / self.sim_fps) // self.mujoco_dt)
        self.metadata["render_fps"] = int(np.round(1 / self.mujoco_dt / frame_skip))

        MujocoEnv.__init__(
            self,
            model_path=str(Path(model_dir).absolute() / f"{self.model_file}.xml"),
            frame_skip=frame_skip,
            render_mode="rgb_array",  # Internal render mode for getting frames
            observation_space=observation_space,
            width=self.render_width,
            height=self.render_height,
        )


        # Ensure action space uses float32
        self.action_space = Box(
            self.action_space.low.astype(np.float32),
            self.action_space.high.astype(np.float32),
            dtype=np.float32,
        )

        # --- Post-Initialization Calculations ---
        self.steps_per_cycle = int(
            self.dt_per_cycle / (self.mujoco_dt * self.frame_skip)
        )

        self._push_duration: int = int(config["push_duration"] / self.dt)


        # Precompute Von Mises values for efficiency
        phis = np.linspace(0, 1, self.steps_per_cycle, endpoint=False)
        self.von_mises_values_swing = np.array(
            [
                p_between_von_mises(
                    a=self.a_swing, b=self.b_swing, kappa=self.kappa, x=p
                )
                for p in phis
            ],
            dtype=np.float32,
        )
        self.von_mises_values_stance = np.array(
            [
                p_between_von_mises(
                    a=self.a_stance, b=self.b_stance, kappa=self.kappa, x=p
                )
                for p in phis
            ],
            dtype=np.float32,
        )

        # Adaptive normalization ranges for reward components
        self.exponents_ranges = {
            "q_vx": (0.0, self.x_cmd_vel),
            "q_vy": (0.0, self.y_cmd_vel),
            "q_left_frc": (0.0, 2 * gravity * mass),
            "q_right_frc": (0.0, 2 * gravity * mass),
            "q_left_spd": (0.0, np.sqrt(self.x_cmd_vel**2 + self.y_cmd_vel**2)),
            "q_right_spd": (0.0, np.sqrt(self.x_cmd_vel**2 + self.y_cmd_vel**2)),
            "q_action": (0.0, 1.0),
            "q_pelvis_acc": (0.0, 10.0),
            "q_orientation": (0.0, 2 * np.sqrt(2)), # norm is between (0 and sqr(2 + 2 + 2 + 2))
            "q_torque": (0.0, np.max(self.action_space.high)),
        }

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def quat_to_rpy(
        quaternion: "npt.NDArray[np.float32]", radians: bool = True
    ) -> "npt.NDArray[np.float32]":
        """Converts a quaternion (w, x, y, z) to roll, pitch, yaw Euler angles."""
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        # Handle gimbal lock: sinp approaches +/- 1
        # Use np.clip to prevent arcsin domain errors due to floating point inaccuracies
        if np.abs(sinp) >= 1:
            # Assign pi/2 or -pi/2 directly
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        rpy = np.array([roll, pitch, yaw], dtype=np.float32)

        return rpy if radians else np.degrees(rpy)

    @property
    def total_simulation_time(self):
        return self.steps * self.dt

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_pelvis_z_range

        self.done_n = 0.0

        self.isdone = "not done"

        if (self.contact and self.data.xpos[PELVIS, 2] > max_z) or self.data.xpos[
            PELVIS, 2
        ] < min_z:
            self.isdone = "Pelvis not in range"
            self.done_n = 1.0

        if self.total_simulation_time > self._max_sim_time:
            self.isdone = "Max steps reached"
            self.done_n = 2.0

        if (
            not abs(self.data.xpos[LEFT_FOOT, 0] - self.data.xpos[RIGHT_FOOT, 0])
            < self._healthy_feet_distance_x
        ):
            self.isdone = "Feet distance out of range along x-axis"
            self.done_n = 3.0

        if (
            not 0.0
            < self.data.xpos[RIGHT_FOOT, 1] - self.data.xpos[LEFT_FOOT, 1]
            < self._healthy_feet_distance_y
        ):
            self.isdone = "Feet distance out of range along y-axis"

        if (
            not abs(self.data.xpos[LEFT_FOOT, 2] - self.data.xpos[RIGHT_FOOT, 2])
            < self._healthy_feet_distance_z
        ):
            self.isdone = "Feet distance out of range along z-axis"
            self.done_n = 4.0

        if (
            self.contact
            and self.data.xpos[LEFT_FOOT, 2] >= self._healthy_feet_height
            and self.data.xpos[RIGHT_FOOT, 2] >= self._healthy_feet_height
            and np.linalg.norm(self.contact_force_left_foot) < 0.01
            and np.linalg.norm(self.contact_force_right_foot) < 0.01
        ):
            self.isdone = "Both Feet not on the ground"
            self.done_n = 5.0

        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[PELVIS, 2] - self.data.xpos[LEFT_FOOT, 2]
        ):
            self.isdone = "Left foot too close to pelvis"
            self.done_n = 6.0

        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[PELVIS, 2] - self.data.xpos[RIGHT_FOOT, 2]
        ):
            self.isdone = "Right foot too close to pelvis"
            self.done_n = 7.0

        pelvis_rpy = self.quat_to_rpy(self.data.sensordata[16:20])
        rpy_diff = np.abs(mod(pelvis_rpy - self.init_rpy, np.pi))
        if rpy_diff[0] > self.max_roll:
            self.isdone = "Roll too high"
            self.done_n = 8.0
        if rpy_diff[1] > self.max_pitch:
            self.isdone = "Pitch too high"
            self.done_n = 9.0
        if rpy_diff[2] > self.max_yaw:
            self.isdone = "Yaw too high"
            self.done_n = 10.0

        return self.isdone == "not done"

    @property
    def terminated(self):
        terminated = (
            (not self.is_healthy)
            if (self._terminate_when_unhealthy)  # or self.steps > MAX_STEPS)
            else False
        )
        return terminated

    @property
    def truncated(self):
        return self.total_simulation_time > self._max_sim_time

    @property
    def sensor_data(self) -> dict[str, np.ndarray]:
        return {
            group: np.array(
                [self.data.sensor(s).data for s in sensors[group]]
            ).flatten()
            for group in sensors.keys()
        }

    @staticmethod
    def _get_symmetric_obs(obs: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        symmetric_obs = deepcopy(obs)

        # actuatorpos - swap left and right actuators
        symmetric_obs[0:5] = obs[5:10]
        symmetric_obs[5:10] = obs[0:5]

        # jointpos - swap left and right joints
        symmetric_obs[10:13] = obs[13:16]
        symmetric_obs[13:16] = obs[10:13]

        # pelvis quaternion symmetric along the sagittal plane (xz-plane)
        # For a quaternion [w,x,y,z], the symmetry transformation is [w,-x,y,-z]
        symmetric_obs[17] = -obs[17]  # x component
        symmetric_obs[19] = -obs[19]  # z component

        # pelvis angular velocity - symmetric along sagittal plane
        symmetric_obs[20] = -obs[20]  # x component
        symmetric_obs[22] = -obs[22]  # z component

        # pelvis linear acceleration - symmetric along sagittal plane (y direction flips)
        symmetric_obs[24] = -obs[24]  # y component

        # Magnetometer - symmetric along sagittal plane
        symmetric_obs[27] = -obs[27]  # y component

        # command - y velocity flips sign when mirroring
        symmetric_obs[30] = -obs[30]  # y command

        # contact forces - swap left and right foot forces
        symmetric_obs[31:34] = obs[34:37]  # left foot forces become right foot forces
        symmetric_obs[34:37] = obs[31:34]  # right foot forces become left foot forces

        # clock signal - phase shift of half cycle for symmetric gait
        # Use negative values for sin and cos to represent phase shift of π
        symmetric_obs[37] = -obs[37]  # sin component
        symmetric_obs[38] = -obs[38]  # cos component

        return symmetric_obs

    @staticmethod
    def symmetric_action(action):
        return np.array([*action[5:], *action[:5]])

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def _normalize_reward(
        rewards: "npt.NDArray[np.float32]", reward_coeffs: "npt.NDArray[np.float32]"
    ) -> float:
        return np.sum(rewards * reward_coeffs) / np.sum(reward_coeffs)

    # step in time
    def step(
        self, action: "npt.NDArray[np.float32]"
    ) -> tuple["npt.NDArray[np.float32]", float, bool, bool, dict[str, Any]]:
        if self.symmetric_turn and self.symmetric_regulation:
            act = self.symmetric_action(action)
        else:
            act = action

        self.do_simulation(act, self.frame_skip)

        # Feet Contact Forces
        contacts = [contact.geom2 for contact in self.data.contact]
        self.contact_force_left_foot = np.zeros(6)
        self.contact_force_right_foot = np.zeros(6)

        if LEFT_CONTACT_IDX in contacts:
            m.mj_contactForce(  # type: ignore
                self.model,
                self.data,
                contacts.index(LEFT_CONTACT_IDX),
                self.contact_force_left_foot,
            )
        if RIGHT_CONTACT_IDX in contacts:
            m.mj_contactForce(  # type: ignore
                self.model,
                self.data,
                contacts.index(RIGHT_CONTACT_IDX),
                self.contact_force_right_foot,
            )

        # check if cassie hit the ground with feet
        if (
            self.data.xpos[LEFT_FOOT, 2] < 0.12
            or self.data.xpos[RIGHT_FOOT, 2] < 0.12
            or np.linalg.norm(self.contact_force_left_foot) > 100
            or np.linalg.norm(self.contact_force_right_foot) > 100
        ):
            self.contact = True

        terminated = self.terminated
        reward, metrics = self._compute_reward(act)

        self._set_obs()
        self.symmetric_turn = not self.symmetric_turn
        if self.symmetric_turn and self.symmetric_regulation:
            observation = self._get_symmetric_obs(self.obs)
        else:
            observation = self.obs

        self.steps += 1
        self.phi += 1.0 / self.steps_per_cycle
        self.phi = self.phi % 1

        self.previous_action = act

        info = {}

        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        info["other_metrics"] = metrics

        # Push the robot
        random_force_xy = np.random.uniform(
            -self._force_max_norm, self._force_max_norm, size=2
        )
        full_force = np.concatenate([random_force_xy, np.array([0.0] * 4)])

        sampled = np.random.uniform(0, 1)
        if sampled < self._push_prob and self._pushing == -1:
            self._pushing = 0

        if (
            -1 < self._pushing < self._push_duration
        ):  # Push for self._push_duration * frame_skip * 0.002 seconds(0.002 is the time step of the simulation)
            self.data.xfrc_applied[PELVIS] = full_force
            self._pushing += 1
        else:
            self._pushing = -1

        self.obs = observation
        return (
            observation,
            reward,
            terminated,
            self.truncated,
            info,
        )

    def render(self) -> "bool | npt.NDArray[np.uint8]":
        frame = super().render()

        if self.local_render_mode == "rgb_array":
            return frame
        else:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            initial_width = int(720 * aspect_ratio)
            cv2.imshow("Cassie", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.resizeWindow("Cassie", initial_width, 720)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' key or ESC key
                self.is_running = False
                cv2.destroyAllWindows()
                return None
            elif key == ord("r"):  # 'r' key to reset window size
                cv2.resizeWindow("Cassie", 800, 600)

            return True

    # resets the simulation
    def reset_model(self, seed: int = None) -> "npt.NDArray[np.float32]":
        # set seed
        np.random.seed(seed)

        self.done_n = 0.0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10, dtype=np.float32)
        self.contact_force_left_foot = np.zeros(6, dtype=np.float32)
        self.contact_force_right_foot = np.zeros(6, dtype=np.float32)

        # self.phi = np.random.randint(0, self.steps_per_cycle) / self.steps_per_cycle
        self.phi = 0.0
        self.steps = 0

        self.contact = False

        self.symmetric_turn = bool(np.random.choice([True, False]))
        self.command = np.array([self.x_cmd_vel, self.y_cmd_vel], dtype=np.float32)

        self._pushing = -1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.data.xfrc_applied[PELVIS] = np.zeros(6)  # Reset external forces

        # Ensure data is synced before accessing sensors
        m.mj_forward(self.model, self.data)  # type: ignore

        # Store initial orientation *after* setting state and running mj_forward
        self.init_rpy = self.quat_to_rpy(self.sensor_data["framequat"])

        self._set_obs()

        if self.obs is None:
            raise RuntimeError("Observation was not set during reset.")

        return (
            self._get_symmetric_obs(self.obs)
            if self.symmetric_turn and self.symmetric_regulation
            else self.obs
        )

    def normalize_reward(
        self, rewards: Dict[str, float], reward_coeffs: Dict[str, float]
    ) -> float:
        """Normalizes the reward components based on their coefficients."""
        # Ensure order consistency (though dict order is guaranteed in Python 3.7+)
        reward_keys = sorted(rewards.keys())
        reward_values = np.array([rewards[k] for k in reward_keys], dtype=np.float32)
        # Filter and align coefficients
        coeffs_values = np.array(
            [reward_coeffs[k] for k in reward_keys if k in reward_coeffs],
            dtype=np.float32,
        )

        if len(reward_values) != len(coeffs_values):
            log.warning(
                f"Mismatch between reward components ({len(reward_values)}) and coefficients ({len(coeffs_values)})"
            )
            # Attempt to match based on available keys - this might indicate a config error
            matched_keys = [k for k in reward_keys if k in reward_coeffs]
            reward_values = np.array(
                [rewards[k] for k in matched_keys], dtype=np.float32
            )
            coeffs_values = np.array(
                [reward_coeffs[k] for k in matched_keys], dtype=np.float32
            )

        if np.sum(coeffs_values) == 0:
            log.warning(
                "Sum of reward coefficients is zero, returning raw sum of rewards."
            )
            return np.sum(reward_values)

        # Use the Numba-optimized function
        return self._normalize_reward(reward_values, coeffs_values)

    def _set_obs(self):
        """Constructs the observation vector from sensor data and internal state."""
        # Calculate clock signal
        clock_signal = np.array(
            [np.sin((2 * np.pi * self.phi)), np.cos((2 * np.pi * self.phi))]
        )

        # Get sensor data
        sensor_readings = self.sensor_data

        # Assemble observation vector
        self.obs = np.concatenate(
            [
                sensor_readings["actuatorpos"],  # 0-9
                sensor_readings["jointpos"],  # 10-15
                sensor_readings["framequat"],  # 16-19
                sensor_readings["gyro"],  # 20-22
                sensor_readings["accelerometer"],  # 23-25
                sensor_readings["magnetometer"],  # 26-28
                self.command,  # 29-30
                self.contact_force_left_foot[:3],  # 31-33 (Use only first 3 components)
                self.contact_force_right_foot[
                    :3
                ],  # 34-36 (Use only first 3 components)
                clock_signal,  # 37-38
            ]
        ).astype(np.float32)  # Ensure float32 type

    def _get_obs(self):
        """Returns the current observation."""
        if self.obs is None:
            # This might happen if called before the first reset or step
            log.warning(
                "Attempted to get observation before it was set. Performing reset."
            )
            self.reset()  # Call reset to ensure obs is initialized
        # We assume self.reset() correctly sets self.obs
        return self.obs  # type: ignore

    def _normalize_quantity(self, name: str, q: float) -> float:
        """
        Applies exponential moving average to update normalization range
        and returns the normalized quantity.
        """
        k = 2.0  # Smoothing factor for EMA
        current_min, current_max = self.exponents_ranges[name]

        # Update range using EMA
        new_min = ((k - 1) * current_min + max(q, current_min)) / k

        new_min = current_min
        new_max = ((k - 1) * current_max + min(q, current_max)) / k
        new_max = current_max
        # Ensure min <= max
        new_min = min(new_min, new_max)
        new_max = max(new_min, new_max) + 1e-6

        self.exponents_ranges[name] = (new_min, new_max)

        # Normalize the quantity using the updated range
        range_size = new_max - new_min
        # Prevent division by zero or near-zero
        if range_size < 1e-6:
            return 0.5  # Return midpoint if range is collapsed
        else:
            normalized_q = (q - new_min) / range_size
            # Clamp result to [0, 1] as normalization might slightly exceed bounds due to EMA lag
            return np.clip(normalized_q, 0.0, 1.0)

    # computes the reward
    def _compute_reward(
        self, action: "npt.NDArray[np.float32]"
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Computes the reward components and the final reward value."""

        # --- Phase-Dependent Reward Modulation ---

        # Helper functions using precomputed Von Mises values
        def i_swing_frc(phi):
            """Importance of low force during swing phase."""
            idx = int(round((phi % 1.0) * self.steps_per_cycle)) % self.steps_per_cycle
            return self.von_mises_values_swing[idx]

        def i_stance_spd(phi):
            """Importance of low speed during stance phase."""
            idx = int(round((phi % 1.0) * self.steps_per_cycle)) % self.steps_per_cycle
            return self.von_mises_values_stance[idx]

        def c_frc(phi):
            """Coefficient for force reward based on phase."""
            return c_swing_frc * i_swing_frc(phi)

        def c_spd(phi):
            """Coefficient for speed reward based on phase."""
            return c_stance_spd * i_stance_spd(phi)

        # Extract necessary simulation data
        qvel = self.data.qvel.copy()  # Use .copy() to avoid modifying internal data
        pelvis_quat = self.sensor_data["framequat"]
        pelvis_ang_vel = self.sensor_data["gyro"]

        # --- Calculate Reward Components (q-values, exponential form) ---

        # Velocity Tracking (Pelvis X/Y Velocity)

        raw_quantities = {
            "q_vx": abs(qvel[0] - self.command[0]),
            "q_vy": abs(qvel[1] - self.command[1]),
            "q_left_frc": np.linalg.norm(self.contact_force_left_foot),
            "q_right_frc": np.linalg.norm(self.contact_force_right_foot),
            "q_left_spd": abs(qvel[12]),
            "q_right_spd": abs(qvel[19]),
            "q_action": action_dist(
                action.reshape(1, -1),  # Reshape for function expected input
                self.previous_action.reshape(1, -1),
                self.action_space.high,
                self.action_space.low,
            )[0],
            "q_pelvis_acc": np.linalg.norm(pelvis_ang_vel),
            "q_torque": np.linalg.norm(action),
            "q_orientation": np.linalg.norm(pelvis_quat - FORWARD_QUARTERNIONS),
        }

        normalized_quantities = {
            k: self._normalize_quantity(k, v) for k, v in raw_quantities.items()
        }

        after_exponential = {
            k: np.exp(-OMEGA * v) for k, v in normalized_quantities.items()
        }

        # --- Combine Reward Components ---

        # Command Following Reward
        r_cmd = (
            1.0 * after_exponential["q_vx"]
            + 0.8 * after_exponential["q_vy"]
            + 0.2 * after_exponential["q_orientation"]
        ) / (1.0 + 0.8 + 0.2)

        # Smoothness Reward
        r_smooth = (
            1.0 * after_exponential["q_action"]
            + 0.5 * after_exponential["q_torque"]
            + 0.5 * after_exponential["q_pelvis_acc"]
        ) / (1.0 + 0.5 + 0.5)

        # Bipedal Locomotion Reward (Phase-modulated foot forces and speeds)
        r_biped = 0.0
        r_biped += (
            c_frc(self.phi + THETA_LEFT) * after_exponential["q_left_frc"]
        )  # Penalize left foot force during its swing phase
        r_biped += (
            c_frc(self.phi + THETA_RIGHT) * after_exponential["q_right_frc"]
        )  # Penalize right foot force during its swing phase
        r_biped += (
            c_spd(self.phi + THETA_LEFT) * after_exponential["q_left_spd"]
        )  # Penalize left foot speed during its stance phase
        r_biped += (
            c_spd(self.phi + THETA_RIGHT) * after_exponential["q_right_spd"]
        )  # Penalize right foot speed during its stance phase
        r_biped /= 2.0  # Average the four components

        # Store individual reward components
        rewards: Dict[str, float] = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
        }

        # Calculate final weighted reward using coefficients from config
        total_reward = self.normalize_reward(rewards, self.reward_coeffs)

        # Add bias term (often used as a small penalty per step to encourage efficiency)
        total_reward += self.reward_coeffs.get("bias", -0.01)

        # --- Metrics for Logging/Debugging ---
        metrics = {
            "raw_quantities": raw_quantities,
            "normalized_quantities": normalized_quantities,
            "after_exponential": after_exponential,
            "rewards": rewards,
            "total_reward": total_reward,
        }

        return total_reward, metrics