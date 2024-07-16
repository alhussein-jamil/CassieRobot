import logging as log
from typing import Any, TYPE_CHECKING

import cv2
import mujoco as m
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from pathlib import Path
import numba as nb
from collections import OrderedDict

from .functions import action_dist, p_between_von_mises

from .constants import (
    PELVIS,
    LEFT_FOOT,
    RIGHT_FOOT,
    pos_index,
    vel_index,
    left_foot_force_idx,
    right_foot_force_idx,
    FORWARD_QUARTERNIONS,
    THETA_LEFT,
    THETA_RIGHT,
    c_stance_spd,
    c_swing_frc,
    sensors,
    full_symmetry_matrix,
    mass,
    gravity,
)

if TYPE_CHECKING:
    import numpy.typing as npt

log.basicConfig(level=log.INFO)

DEFAULT_CONFIG = {
    "symmetric_regulation": True,
    "steps_per_cycle": 30,
    "r": 1.0,
    "kappa": 25,
    "x_cmd_vel": 1.5,
    "y_cmd_vel": 0,
    "terminate_when_unhealthy": True,
    "max_simulation_steps": 400,
    "pelvis_height": [0.75, 1.25],
    "feet_distance_x": 1.0,
    "feet_distance_y": 0.5,
    "feet_distance_z": 0.5,
    "feet_pelvis_height": 0.3,
    "feet_height": 0.6,
    "model": "cassie",
    "render_mode": "rgb_array",
    "reset_noise_scale": 0.01,
    "force_max_norm": 0.0,
    "push_freq": 0,
    "push_duration": 0,
    "bias": -0.01,
    "r_biped": 4.0,
    "r_cmd": 3.0,
    "r_smooth": 1.0,
    "r_alternate": 4.0,
    "r_symmetric": 2.0,
    "is_training": True,
    "max_roll": 0.2,
    "max_pitch": 0.2,
    "max_yaw": 0.2,
}


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self, env_config: dict[str, Any], model_dir: str | Path = "assets/model"
    ):
        self.symmetric_regulation = env_config.get(
            "symmetric_regulation", DEFAULT_CONFIG["symmetric_regulation"]
        )
        self._terminate_when_unhealthy = env_config.get(
            "terminate_when_unhealthy", DEFAULT_CONFIG["terminate_when_unhealthy"]
        )
        self._healthy_pelvis_z_range = env_config.get(
            "pelvis_height", DEFAULT_CONFIG["pelvis_height"]
        )
        self._healthy_feet_distance_x = env_config.get(
            "feet_distance_x", DEFAULT_CONFIG["feet_distance_x"]
        )
        self._healthy_feet_distance_y = env_config.get(
            "feet_distance_y", DEFAULT_CONFIG["feet_distance_y"]
        )
        self._healthy_feet_distance_z = env_config.get(
            "feet_distance_z", DEFAULT_CONFIG["feet_distance_z"]
        )
        self._healthy_dis_to_pelvis = env_config.get(
            "feet_pelvis_height", DEFAULT_CONFIG["feet_pelvis_height"]
        )
        self._healthy_feet_height = env_config.get(
            "feet_height", DEFAULT_CONFIG["feet_height"]
        )
        self._max_steps = env_config.get(
            "max_simulation_steps", DEFAULT_CONFIG["max_simulation_steps"]
        )
        self.steps_per_cycle = env_config.get(
            "steps_per_cycle", DEFAULT_CONFIG["steps_per_cycle"]
        )

        self.obs = np.zeros(41)
        self.max_roll = env_config.get("max_roll", DEFAULT_CONFIG["max_roll"])
        self.max_pitch = env_config.get("max_pitch", DEFAULT_CONFIG["max_pitch"])
        self.max_yaw = env_config.get("max_yaw", DEFAULT_CONFIG["max_yaw"])
        self.training = env_config.get("is_training", DEFAULT_CONFIG["is_training"])
        self.r = env_config.get("r", DEFAULT_CONFIG["r"])
        self.a_swing = 0.0
        self.a_stance = self.r
        self.b_swing = self.a_stance
        self.b_stance = 1.0
        self.kappa = env_config.get("kappa", DEFAULT_CONFIG["kappa"])
        self.x_cmd_vel = env_config.get("x_cmd_vel", DEFAULT_CONFIG["x_cmd_vel"])
        self.y_cmd_vel = env_config.get("y_cmd_vel", DEFAULT_CONFIG["y_cmd_vel"])
        self.model_file = env_config.get("model", "cassie")

        # Massive Speedup by storing the von mises values and reusing them
        phis = np.linspace(0, 1, self.steps_per_cycle, endpoint=False)
        self.von_mises_values_swing = np.array(
            [
                p_between_von_mises(
                    a=self.a_swing, b=self.b_swing, kappa=self.kappa, x=p
                )
                for p in phis
            ]
        )
        self.von_mises_values_stance = np.array(
            [
                p_between_von_mises(
                    a=self.a_stance, b=self.b_stance, kappa=self.kappa, x=p
                )
                for p in phis
            ]
        )

        # dictionary of keys containing r_
        self.reward_coeffs = {
            k: v
            for k, v in env_config.items()
            if k.startswith("r_") or k.startswith("bias")
        }
        if len(self.reward_coeffs) == 0:
            self.reward_coeffs = {
                k: v
                for k, v in DEFAULT_CONFIG.items()
                if k.startswith("r_") or k.startswith("bias")
            }

        self._reset_noise_scale = env_config.get(
            "reset_noise_scale", DEFAULT_CONFIG["reset_noise_scale"]
        )
        self._force_max_norm = env_config.get(
            "force_max_norm", DEFAULT_CONFIG["force_max_norm"]
        )
        self._push_freq = env_config.get("push_freq", DEFAULT_CONFIG["push_freq"])
        self._push_duration = env_config.get(
            "push_duration", DEFAULT_CONFIG["push_duration"]
        )

        self._push_probability = 1.0 / self._push_freq if self._push_freq > 0 else 0.0
        self._pushing = -1

        self.phi, self.steps = 0, 0

        self.previous_action = torch.zeros(10)

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
        self.observation_space = Box(
            low=np.concatenate([x.low for x in observation_space_dict.values()]),
            high=np.concatenate([x.high for x in observation_space_dict.values()]),
        )
        self.metadata["render_fps"] = 2000 // env_config.get("frame_skip", 40)
        MujocoEnv.__init__(
            self,
            model_path=str(Path(model_dir).absolute() / f"{self.model_file}.xml"),
            frame_skip=env_config.get("frame_skip", 40),
            render_mode="rgb_array",
            observation_space=self.observation_space,
            width=env_config.get("width", 1920),
            height=env_config.get("height", 1080),
        )
        self.local_render_mode = env_config.get("render_mode", "rgb_array")
        self.exponents_ranges = dict(
            q_vx=(0.0, self.x_cmd_vel),
            q_vy=(0.0, self.y_cmd_vel),
            q_frc_left=(0.0, gravity * mass),
            q_frc_right=(0.0, gravity * mass),
            q_spd_left=(0.0, np.real(np.sqrt(self.x_cmd_vel**2 + self.y_cmd_vel**2))),
            q_spd_right=(0.0, np.real(np.sqrt(self.x_cmd_vel**2 + self.y_cmd_vel**2))),
            q_action=(0.0, 1.0),
            pelvis_orientation=(0.0, 1.0),
            q_feet_orientation=(0.0, 1.0),
            q_torque=(0.0, np.max(self.action_space.high)),
            q_pelvis_acc=(0.0, 1.0),
            q_symmetric=(0.0, 1.0),
        )
        self.exponents_ranges = { k: (v[0], max(v[1], 0.1)) for k, v in self.exponents_ranges.items() }

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def quat_to_rpy(
        quaternion: "npt.NDArray[np.float32]", radians: bool = True
    ) -> "npt.NDArray[np.float32]":
        q = quaternion
        yaw = np.arctan2(
            2.0 * (q[1] * q[2] + q[3] * q[0]),
            q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2],
        )
        pitch = np.arcsin(-2.0 * (q[0] * q[2] - q[3] * q[1]))
        roll = np.arctan2(
            2.0 * (q[0] * q[1] + q[3] * q[2]),
            q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2],
        )
        return (
            np.array([roll, pitch, yaw], dtype=np.float32)
            if radians
            else np.degrees(np.array([roll, pitch, yaw], dtype=np.float32))
        )

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

        if self.steps > self._max_steps:
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
        rpy_diff = np.abs(pelvis_rpy - self.init_rpy)
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
        return self.steps > self._max_steps

    def _set_obs(self):
        p = np.array(
            [np.sin((2 * np.pi * (self.phi))), np.cos((2 * np.pi * (self.phi)))]
        )
        self.obs = np.concatenate(
            [
                np.concatenate(list(self.sensors.values())),  # 0 - 28
                self.command,  # 29 - 30
                np.concatenate(
                    [
                        self.contact_force_left_foot[:3],
                        self.contact_force_right_foot[:3],
                    ]
                ),  # 31 - 36
                np.array([*p]),  # 37 - 38
            ]
        )

    @staticmethod
    def _get_symmetric_obs(obs: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        return np.dot(full_symmetry_matrix, obs)

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
        # self.symmetric_turn = self.phi < 0.5
        if self.symmetric_turn and self.symmetric_regulation:
            act = self.symmetric_action(action)
        else:
            act = action

        # act = self.denormalize_actions(act)
        self.do_simulation(act, self.frame_skip)
        terminated = self.terminated
        reward, rewards, metrics = self._compute_reward(act)  # , sym_act)

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
        info["custom_rewards"] = rewards

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
        if sampled < self._push_probability and self._pushing == -1:
            self._pushing = 0

        if (
            -1 < self._pushing < self._push_duration
        ):  # Push for self._push_duration * frame_skip * 0.002 seconds(0.002 is the time step of the simulation)
            self.data.xfrc_applied[PELVIS] = full_force
            self._pushing += 1
        else:
            self._pushing = -1

        return observation, reward, terminated, self.truncated, info

    # resets the simulation
    def reset_model(self, seed: int = None) -> "npt.NDArray[np.float32]":

        # set seed
        np.random.seed(seed)
        self.done_n = 0.0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10)
        self.contact_force_left_foot = np.zeros(6)
        self.contact_force_right_foot = np.zeros(6)
        self.contact = False

        # self.phi = np.random.randint(0, self.steps_per_cycle) / self.steps_per_cycle
        self.phi = 0
        self.steps = 0

        self.symmetric_turn = np.random.choice([True, False])
        self.command = np.array([self.x_cmd_vel, self.y_cmd_vel])

        self._pushing = -1
        self.data.xfrc_applied[PELVIS] = np.zeros(6)

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.init_rpy = self.quat_to_rpy(FORWARD_QUARTERNIONS)

        self._set_obs()

        return (
            self._get_symmetric_obs(self.obs)
            if self.symmetric_turn and self.symmetric_regulation
            else self.obs
        )

    def normalize_reward(
        self, rewards: dict[str, float], reward_coeffs: dict[str, float]
    ):
        reward_values = np.array([rewards[k] for k in rewards.keys()])
        reward_coeffs_values = np.array(
            [reward_coeffs[k] for k in rewards.keys() if k in reward_coeffs.keys()]
        )
        return self._normalize_reward(reward_values, reward_coeffs_values)

    @property
    def sensors(self):
        return {
            group: np.array(
                [self.data.sensor(s).data for s in sensors[group]]
            ).flatten()
            for group in sensors.keys()
        }

    def render(self):
        if self.local_render_mode == "rgb_array":
            return super().render()
        else:
            frame = super().render()
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

            return frame

    def _normalize_quantity(self, name, q):
        k = 5.0
        # Update the range
        self.exponents_ranges[name] = (
            (
                (k - 1) * self.exponents_ranges[name][0]
                + min(self.exponents_ranges[name][0], q)
            )
            / k,
            (
                (k - 1) * self.exponents_ranges[name][1]
                + max(self.exponents_ranges[name][1], q)
            )
            / k,
        )

        # Normalize the quantity
        return (q - self.exponents_ranges[name][0]) / (
            self.exponents_ranges[name][1] - self.exponents_ranges[name][0] + 1e-6
        )

    # computes the reward
    def _compute_reward(self, action: "npt.NDArray[np.float32]"):
        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos = qpos[pos_index]
        qvel = qvel[vel_index]

        # # Feet Contact Forces
        contacts = [contact.geom2 for contact in self.data.contact]
        self.contact_force_left_foot = np.zeros(6)

        self.contact_force_right_foot = np.zeros(6)

        if left_foot_force_idx in contacts:
            m.mj_contactForce(  # type: ignore
                self.model,
                self.data,
                contacts.index(left_foot_force_idx),
                self.contact_force_left_foot,
            )
        if right_foot_force_idx in contacts:
            m.mj_contactForce(  # type: ignore
                self.model,
                self.data,
                contacts.index(right_foot_force_idx),
                self.contact_force_right_foot,
            )

        # check if cassie hit the ground with feet
        if (
            # self.data.xpos[LEFT_FOOT, 2] < 0.08
            # or self.data.xpos[RIGHT_FOOT, 2] < 0.08
            np.linalg.norm(self.contact_force_left_foot) > 100
            or np.linalg.norm(self.contact_force_right_foot) > 100
        ):
            self.contact = True

        # Some metrics to be used in the reward function
        q_vx = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_vx",
                np.linalg.norm(np.array([qvel[0]]) - np.array([self.command[0]])),
            )
        )
        q_vy = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_vy",
                np.linalg.norm(np.array([qvel[1]]) - np.array([self.command[1]])),
            )
        )

        q_left_frc = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_frc_left", np.linalg.norm(self.contact_force_left_foot)
            )
        )
        q_right_frc = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_frc_right", np.linalg.norm(self.contact_force_right_foot)
            )
        )
        q_left_spd = np.exp(
            -6.0 * self._normalize_quantity("q_spd_left", np.linalg.norm(qvel[12]))
        )
        q_right_spd = np.exp(
            -6.0 * self._normalize_quantity("q_spd_right", np.linalg.norm(qvel[19]))
        )
        q_action_diff = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_action",
                float(
                    action_dist(
                        a=np.array(action).reshape(1, -1),
                        b=np.array(self.previous_action).reshape(1, -1),
                        action_space_low=self.action_space.low,
                        action_space_high=self.action_space.high,
                    )
                ),
            )
        )
        q_orientation = np.exp(
            -6.0
            * self._normalize_quantity(
                "pelvis_orientation",
                np.linalg.norm(
                    self.data.sensor("pelvis-orientation").data - FORWARD_QUARTERNIONS
                ),
            )
        )
        q_torque = np.exp(
            -6.0 * self._normalize_quantity("q_torque", np.linalg.norm(action))
        )
        q_pelvis_acc = np.exp(
            -6.0
            * self._normalize_quantity(
                "q_pelvis_acc",
                (np.linalg.norm(self.data.sensor("pelvis-angular-velocity").data)),
            )
        )

        def i_swing_frc(phi):
            index = round(phi * self.steps_per_cycle) % self.steps_per_cycle
            return self.von_mises_values_swing[index]

        def i_stance_spd(phi):
            index = round(phi * self.steps_per_cycle) % self.steps_per_cycle
            return self.von_mises_values_stance[index]

        def c_frc(phi):
            return c_swing_frc * i_swing_frc(phi)

        def c_spd(phi):
            return c_stance_spd * i_stance_spd(phi)

        # Compute the rewards

        ## Compute Command Reward
        r_cmd = (+1.0 * q_vx + 0.8 * q_vy + 0.2 * q_orientation) / (1.0 + 0.8 + 0.2)

        ## Compute Smoothness Reward
        r_smooth = (1.0 * q_action_diff + 0.5 * q_torque + 0.5 * q_pelvis_acc) / (
            1.0 + 0.5 + 0.5
        )

        ## Compute Bipedal Reward
        r_biped = 0
        r_biped += c_frc(self.phi + THETA_LEFT) * q_left_frc
        r_biped += c_frc(self.phi + THETA_RIGHT) * q_right_frc
        r_biped += c_spd(self.phi + THETA_LEFT) * q_left_spd
        r_biped += c_spd(self.phi + THETA_RIGHT) * q_right_spd
        r_biped /= 2.0

        rewards = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
        }
        reward = self.normalize_reward(
            rewards, self.reward_coeffs
        ) + self.reward_coeffs.get("bias", -0.01)
        metrics = {
            "dis_x": self.data.geom_xpos[16, 0],
            "dis_y": self.data.geom_xpos[16, 1],
            "dis_z": self.data.geom_xpos[16, 2],
            "vel_x": self.data.qvel[0],
            "vel_y": self.data.qvel[1],
            "vel_z": self.data.qvel[2],
            "feet_distance_x": abs(
                self.data.xpos[LEFT_FOOT, 0] - self.data.xpos[RIGHT_FOOT, 0]
            ),
            "feet_distance_y": abs(
                self.data.xpos[LEFT_FOOT, 1] - self.data.xpos[RIGHT_FOOT, 1]
            ),
            "feet_distance_z": abs(
                self.data.xpos[LEFT_FOOT, 2] - self.data.xpos[RIGHT_FOOT, 2]
            ),
        }
        return reward, rewards, metrics


class MyCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        episode: EpisodeV2,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        for key in episode._last_infos["agent0"].keys():
            for key2 in episode._last_infos["agent0"][key].keys():
                if key + "_" + key2 not in episode.user_data.keys():
                    episode.user_data[key + "_" + key2] = []
                    episode.hist_data[key + "_" + key2] = []
                episode.user_data[key + "_" + key2].append(
                    episode._last_infos["agent0"][key][key2]
                )
                episode.hist_data[key + "_" + key2].append(
                    episode._last_infos["agent0"][key][key2]
                )

    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        **kwargs,
    ):
        for key in episode._last_infos["agent0"].keys():
            for key2 in episode._last_infos["agent0"][key].keys():
                episode.custom_metrics[key + "_" + key2] = np.mean(
                    episode.user_data[key + "_" + key2]
                )
                episode.hist_data[key + "_" + key2] = episode.user_data[
                    key + "_" + key2
                ]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_postprocess_trajectory(
        self,
        *,
        episode: Episode,
        **kwargs,
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
