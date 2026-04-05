import logging as log
from copy import deepcopy
from typing import Any, Dict, Tuple, TYPE_CHECKING
from pathlib import Path
from collections import OrderedDict

import cv2
import mujoco as m
import numba as nb
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

from .constants import (
    DEFAULT_CONFIG,
    PELVIS,
    LEFT_FOOT,
    RIGHT_FOOT,
    LEFT_CONTACT_IDX,
    RIGHT_CONTACT_IDX,
    sensors,
)
from .functions import p_between_von_mises
from .health import check_health
from .observations import build_observation, get_symmetric_obs, symmetric_action
from .rewards import RewardCalculator

if TYPE_CHECKING:
    import numpy.typing as npt

log.basicConfig(level=log.INFO)


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 0,
    }
    mujoco_dt = 0.0005

    def __init__(
        self,
        env_config: Dict[str, Any] | None = None,
        model_dir: str | Path = "assets/model",
    ):
        config = deepcopy(DEFAULT_CONFIG)
        if env_config:
            config.update(env_config)

        # --- Configuration Parameters ---
        self.symmetric_regulation: str = config["symmetric_regulation"]
        assert self.symmetric_regulation in ["alternate", "random", "none"]
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
        self.r: float = config["r"]
        self.kappa: float = config["kappa"]
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

        self.reward_coeffs = {k: v for k, v in config.items() if k.startswith("r_")}

        # --- State ---
        self.a_swing: float = 0.0
        self.a_stance: float = self.r
        self.b_swing: float = self.a_stance
        self.b_stance: float = 1.0
        self._pushing: int = -1
        self.phi: float = 0.0
        self.steps: int = 0
        self.previous_action: npt.NDArray[np.float32] = np.zeros(10)
        self.command: npt.NDArray[np.float32] = np.array(
            [self.x_cmd_vel, self.y_cmd_vel]
        )
        self.contact: bool = False
        self.symmetric_turn: bool = False
        self.init_rpy: npt.NDArray[np.float32] | None = None
        self.contact_force_left_foot: npt.NDArray[np.float32] = np.zeros(6)
        self.contact_force_right_foot: npt.NDArray[np.float32] = np.zeros(6)
        self.obs: npt.NDArray[np.float32] | None = None

        # Used by health checks (set after check_health call)
        self.done_n: float = 0.0
        self.isdone: str = "not done"

        # --- Observation Space ---
        observation_space_dict = OrderedDict(
            [
                ("actuatorpos", Box(-180.0, 180.0, shape=(10,))),
                ("jointpos", Box(-180.0, 180.0, shape=(6,))),
                ("framequat", Box(-1.0, 1.0, shape=(4,))),
                ("gyro", Box(-np.inf, np.inf, shape=(3,))),
                ("accelerometer", Box(-np.inf, np.inf, shape=(3,))),
                ("magnetometer", Box(-np.inf, np.inf, shape=(3,))),
                ("command", Box(-np.inf, np.inf, shape=(2,))),
                ("contact_forces", Box(-np.inf, np.inf, shape=(6,))),
                ("clock", Box(-1.0, 1.0, shape=(2,))),
            ]
        )
        _obs_low = np.concatenate([s.low for s in observation_space_dict.values()])
        _obs_high = np.concatenate([s.high for s in observation_space_dict.values()])
        observation_space = Box(low=_obs_low, high=_obs_high, dtype=np.float32)

        # --- MujocoEnv Initialization ---
        frame_skip = int((1.0 / self.sim_fps) // self.mujoco_dt)
        self.metadata["render_fps"] = int(np.round(1 / self.mujoco_dt / frame_skip))

        MujocoEnv.__init__(
            self,
            model_path=str(Path(model_dir).absolute() / f"{self.model_file}.xml"),
            frame_skip=frame_skip,
            render_mode="rgb_array",
            observation_space=observation_space,
            width=self.render_width,
            height=self.render_height,
        )

        self.action_space = Box(
            self.action_space.low.astype(np.float32),
            self.action_space.high.astype(np.float32),
            dtype=np.float32,
        )

        # --- Post-Init ---
        self.steps_per_cycle = int(self.dt_per_cycle / self.dt)
        if self.steps_per_cycle <= 0:
            log.warning(
                "steps_per_cycle calculated as %d, setting to 1", self.steps_per_cycle
            )
            self.steps_per_cycle = 1

        self._push_duration: int = int(config["push_duration"] / self.dt)

        # Precompute Von Mises values
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

        # Initialize reward calculator
        self._reward_calc = RewardCalculator(
            reward_coeffs=self.reward_coeffs,
            x_cmd_vel=self.x_cmd_vel,
            y_cmd_vel=self.y_cmd_vel,
            action_space_high=self.action_space.high,
            action_space_low=self.action_space.low,
            steps_per_cycle=self.steps_per_cycle,
            von_mises_values_swing=self.von_mises_values_swing,
            von_mises_values_stance=self.von_mises_values_stance,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def quat_to_rpy(
        quaternion: "npt.NDArray[np.float32]", radians: bool = True
    ) -> "npt.NDArray[np.float32]":
        """Converts a quaternion (w, x, y, z) to roll, pitch, yaw."""
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        return rpy if radians else np.degrees(rpy)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_simulation_time(self):
        return self.steps * self.dt

    @property
    def sensor_data(self) -> dict[str, np.ndarray]:
        return {
            group: np.array(
                [self.data.sensor(s).data for s in sensors[group]]
            ).flatten()
            for group in sensors.keys()
        }

    @property
    def foot_contact(self):
        return (
            self.data.xpos[LEFT_FOOT, 2] < self._healthy_feet_height
            or self.data.xpos[RIGHT_FOOT, 2] < self._healthy_feet_height
            or np.linalg.norm(self.contact_force_left_foot) > 100
            or np.linalg.norm(self.contact_force_right_foot) > 100
        )

    @property
    def is_healthy(self):
        pelvis_rpy = self.quat_to_rpy(self.data.sensordata[16:20])
        healthy, reason, code = check_health(
            data=self.data,
            pelvis_z_range=self._healthy_pelvis_z_range,
            feet_distance_x=self._healthy_feet_distance_x,
            feet_distance_y=self._healthy_feet_distance_y,
            feet_distance_z=self._healthy_feet_distance_z,
            dis_to_pelvis=self._healthy_dis_to_pelvis,
            feet_height=self._healthy_feet_height,
            max_roll=self.max_roll,
            max_pitch=self.max_pitch,
            max_yaw=self.max_yaw,
            total_sim_time=self.total_simulation_time,
            max_sim_time=self._max_sim_time,
            contact=self.contact,
            foot_in_contact=self.foot_contact,
            init_rpy=self.init_rpy,
            pelvis_rpy=pelvis_rpy,
        )
        self.isdone = reason
        self.done_n = code
        return healthy

    @property
    def terminated(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    @property
    def truncated(self):
        return self.total_simulation_time > self._max_sim_time

    # ------------------------------------------------------------------
    # Symmetric regulation
    # ------------------------------------------------------------------

    def update_symmetric_turn(self):
        if self.symmetric_regulation == "alternate":
            self.symmetric_turn = not self.symmetric_turn
        elif self.symmetric_regulation == "random":
            self.symmetric_turn = np.random.rand() < 0.5
        elif self.symmetric_regulation == "none":
            self.symmetric_turn = False

    # ------------------------------------------------------------------
    # Observation helpers (kept as thin wrappers for backward compat)
    # ------------------------------------------------------------------

    def _set_obs(self):
        self.obs = build_observation(
            sensor_data=self.sensor_data,
            command=self.command,
            contact_force_left_foot=self.contact_force_left_foot,
            contact_force_right_foot=self.contact_force_right_foot,
            phi=self.phi,
        )

    @staticmethod
    def _get_symmetric_obs(obs: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        return get_symmetric_obs(obs)

    @staticmethod
    def symmetric_action(action):
        return symmetric_action(action)

    def _get_obs(self):
        if self.obs is None:
            log.warning("Observation not set; performing reset.")
            self.reset()
        return self.obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: "npt.NDArray[np.float32]"
    ) -> tuple["npt.NDArray[np.float32]", float, bool, bool, dict[str, Any]]:
        act = self.symmetric_action(action) if self.symmetric_turn else action
        self.do_simulation(act, self.frame_skip)

        # Feet contact forces
        self._update_contact_forces()

        if self.foot_contact:
            self.contact = True

        terminated = self.terminated

        # Compute reward via the dedicated calculator
        reward, metrics = self._reward_calc.compute(
            action=act,
            previous_action=self.previous_action,
            qvel=self.data.qvel.copy(),
            command=self.command,
            pelvis_quat=self.sensor_data["framequat"],
            pelvis_ang_vel=self.sensor_data["gyro"],
            left_foot_rpy=self.quat_to_rpy(self.data.xquat[LEFT_FOOT]),
            right_foot_rpy=self.quat_to_rpy(self.data.xquat[RIGHT_FOOT]),
            contact_force_left=self.contact_force_left_foot,
            contact_force_right=self.contact_force_right_foot,
            phi=self.phi,
        )

        self._set_obs()
        self.update_symmetric_turn()
        observation = (
            self._get_symmetric_obs(self.obs) if self.symmetric_turn else self.obs
        )

        self.steps += 1
        self.phi = (self.phi + 1.0 / self.steps_per_cycle) % 1
        self.previous_action = act

        info = self._build_step_info(metrics, reward)

        # Random push perturbation
        self._apply_push()

        self.obs = observation
        return observation, reward, terminated, self.truncated, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self, seed: int = None) -> "npt.NDArray[np.float32]":
        np.random.seed(seed)

        self.done_n = 0.0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10, dtype=np.float32)
        self.contact_force_left_foot = np.zeros(6, dtype=np.float32)
        self.contact_force_right_foot = np.zeros(6, dtype=np.float32)
        self.phi = 0.0
        self.steps = 0
        self.contact = False
        self.command = np.array([self.x_cmd_vel, self.y_cmd_vel], dtype=np.float32)
        self._pushing = -1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.data.xfrc_applied[PELVIS] = np.zeros(6)
        m.mj_forward(self.model, self.data)

        self.init_rpy = self.quat_to_rpy(self.sensor_data["framequat"])
        self._set_obs()

        if self.obs is None:
            raise RuntimeError("Observation was not set during reset.")

        self.update_symmetric_turn()
        return self._get_symmetric_obs(self.obs) if self.symmetric_turn else self.obs

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> "bool | npt.NDArray[np.uint8]":
        frame = super().render()

        if self.local_render_mode == "rgb_array":
            return frame

        height, width = frame.shape[:2]
        aspect_ratio = width / height
        initial_width = int(720 * aspect_ratio)
        cv2.imshow("Cassie", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.resizeWindow("Cassie", initial_width, 720)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            self.is_running = False
            cv2.destroyAllWindows()
            return None
        elif key == ord("r"):
            cv2.resizeWindow("Cassie", 800, 600)

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_contact_forces(self):
        contacts = [contact.geom2 for contact in self.data.contact]
        self.contact_force_left_foot = np.zeros(6)
        self.contact_force_right_foot = np.zeros(6)

        if LEFT_CONTACT_IDX in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(LEFT_CONTACT_IDX),
                self.contact_force_left_foot,
            )
        if RIGHT_CONTACT_IDX in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(RIGHT_CONTACT_IDX),
                self.contact_force_right_foot,
            )

    def _build_step_info(self, metrics: dict, reward: float) -> dict:
        info: dict[str, Any] = {}
        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        if "rewards" in metrics:
            for k, v in metrics["rewards"].items():
                info["custom_metrics"][k] = v
        if "coefficients" in metrics:
            for k, v in metrics["coefficients"].items():
                info["custom_metrics"][k] = v

        exp_keys = ["q_left_frc", "q_right_frc", "q_left_spd", "q_right_spd"]
        if "after_exponential" in metrics:
            for key in exp_keys:
                if key in metrics["after_exponential"]:
                    info["custom_metrics"][f"exp_{key}"] = metrics["after_exponential"][
                        key
                    ]

        info["other_metrics"] = metrics
        return info

    def _apply_push(self):
        random_force_xy = np.random.uniform(
            -self._force_max_norm, self._force_max_norm, size=2
        )
        full_force = np.concatenate([random_force_xy, np.array([0.0] * 4)])

        if np.random.uniform(0, 1) < self._push_prob and self._pushing == -1:
            self._pushing = 0

        if -1 < self._pushing < self._push_duration:
            self.data.xfrc_applied[PELVIS] = full_force
            self._pushing += 1
        else:
            self._pushing = -1
