import logging as log
from copy import deepcopy
from typing import Any, Dict, Tuple, TYPE_CHECKING

import mujoco as m
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from pathlib import Path
from .constants import (
    multiplicators,
    high_action,
    low_action,
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
)
from .functions import action_dist, p_between_von_mises
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt

log.basicConfig(level=log.DEBUG)

DEFAULT_CONFIG = {
    "steps_per_cycle": 30,
    "r": 1.0,
    "kappa": 25,
    "x_cmd_vel": 1.5,
    "y_cmd_vel": 0,
    "z_cmd_vel": 0,
    "terminate_when_unhealthy": True,
    "max_simulation_steps": 400,
    "pelvis_height": [0.6, 1.5],
    "feet_distance_x": 1.0,
    "feet_distance_y": 0.5,
    "feet_distance_z": 0.5,
    "feet_pelvis_height": 0.3,
    "feet_height": 0.6,
    "model": "cassie",
    "render_mode": "rgb_array",
    "reset_noise_scale": 0.01,
    "bias": 1.0,
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

    @staticmethod
    def denormalize_actions(action):
        return (
            action * (high_action - low_action) / 2.0 + (high_action + low_action) / 2.0
        )

    def __init__(
        self, env_config: dict[str, Any], model_dir: str | Path = "assets/model"
    ):
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
        self.z_cmd_vel = env_config.get("z_cmd_vel", DEFAULT_CONFIG["z_cmd_vel"])
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

        self.phi, self.steps = 0, 0

        self.previous_action = torch.zeros(10)

        self.observation_space_dict = dict(
            sensors=Box(np.full(29, -np.inf), np.full(29, np.inf)),
            command=Box(-np.inf, np.inf, shape=(2,)),
            contact_forces=Box(-np.inf, np.inf, shape=(6,)),
            clock=Box(-1.0, 1.0, shape=(2,)),
        )
        self.observation_space = Box(
            low=np.concatenate([x.low for x in self.observation_space_dict.values()]),
            high=np.concatenate([x.high for x in self.observation_space_dict.values()]),
        )
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(40,))
        self.reward_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
        )
        self.metadata["render_fps"] = 2000 // env_config.get("frame_skip", 40)
        MujocoEnv.__init__(
            self,
            model_path=str(Path(model_dir).absolute() / f"{self.model_file}.xml"),
            frame_skip=env_config.get("frame_skip", 40),
            render_mode=env_config.get("render_mode", None),
            observation_space=self.observation_space,
        )
        self.render_mode = "rgb_array"
        self.exponents_ranges = dict(
            q_vx=(0.0, self.x_cmd_vel),
            q_vy=(0.0, self.y_cmd_vel),
            q_vz=(0.0, self.z_cmd_vel),
            q_frc_left=(0.0, 1),
            q_frc_right=(0.0, 1),
            q_spd_left=(0.0, 1),
            q_spd_right=(0.0, 1),
            q_action=(0.0, 1),
            pelvis_orientation=(0.0, 1),
            q_feet_orientation=(0.0, 1),
            q_torque=(0.0, 1),
            q_pelvis_acc=(0.0, 1),
            q_symmetric=(0.0, 1),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def quat_to_rpy(quaternion: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        """Convert quaternion to roll, pitch, yaw angles."""
        x, y, z, w = quaternion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        t2 = np.where(t2 < -1.0, -1.0, t2)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw])

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_pelvis_z_range

        self.done_n = 0.0

        self.isdone = "not done"
        if self.contact and not min_z < self.data.xpos[PELVIS, 2] < max_z:
            self.isdone = "Pelvis not in range"
            self.done_n = 1.0
        if not self.steps <= self._max_steps:
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
    
    @staticmethod
    def _get_symmetric_obs(obs :"npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        symmetric_obs = deepcopy(obs)

        # sensors
        symmetric_obs[0:8] = obs[8:16]
        symmetric_obs[8:16] = obs[0:8]

        # pelvis quaternion symmetric along xoz (w,x,y,z) become (w,-x,y,-z)
        symmetric_obs[17] = -obs[17]
        symmetric_obs[19] = -obs[19]

        # pelvis angular velocity symmetric along xoz
        symmetric_obs[20] = -obs[20]
        symmetric_obs[22] = -obs[22]

        # pelvis acceleration symmetric along xoz
        symmetric_obs[24] = -obs[24]

        # command symmetric along xoz
        symmetric_obs[30] = -obs[30]

        # contact forces symmetric along xoz
        symmetric_obs[31:34] = obs[34:37]
        symmetric_obs[34:37] = obs[31:34]

        # symmetry of the clock
        symmetric_obs[37] = -obs[37]
        symmetric_obs[38] = -obs[38]

        return symmetric_obs
    


    # step in time
    def step(self, action):
        self.symmetric_turn = self.phi < 0.5
        if self.symmetric_turn:
            act = self.symmetric_action(action)
        else:
            act = action
        act = self.denormalize_actions(act)
        self.do_simulation(act, self.frame_skip)

        m.mj_step(self.model, self.data)  # type: ignore
        terminated = self.terminated
        reward, rewards, metrics = self._compute_reward(act)  # , sym_act)

        self._set_obs()
        if self.symmetric_turn:
            observation = self._get_symmetric_obs(self.obs)
        else:
            observation = self.obs

        self.steps += 1
        self.phi += 1.0 / self.steps_per_cycle
        self.phi = self.phi % 1

        self.previous_action = act
        # self.previous_symmetric_action = sym_act

        info = {}
        info["custom_rewards"] = rewards

        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        info["other_metrics"] = metrics

        return observation, reward, terminated, False, info

    # resets the simulation
    def reset_model(self, seed=0):
        # set seed
        np.random.seed(seed)

        m.mj_inverse(self.model, self.data)  # type: ignore

        self.done_n = 0.0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10)
        self.contact_force_left_foot = np.zeros(6)
        self.contact_force_right_foot = np.zeros(6)

        # self.phi = np.random.randint(0, self.steps_per_cycle) / self.steps_per_cycle
        self.phi = 0
        self.steps = 0

        self.contact = False

        self.symmetric_turn = self.phi < 0.5

        self.command = np.array([self.x_cmd_vel, self.y_cmd_vel])

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        self.init_rpy = self.quat_to_rpy(self.data.sensordata[16:20])

        self._set_obs()

        return self._get_symmetric_obs(self.obs) if self.symmetric_turn else self.obs

    @staticmethod
    def symmetric_action(action):
        return np.array([*action[5:], *action[:5]])
    
    @staticmethod
    def normalize_reward(rewards: dict[str, float], reward_coeffs: dict[str, float]):
        return sum([rewards[k] * reward_coeffs[k] for k in rewards.keys()]) / sum(
            [value for key, value in reward_coeffs.items() if key in rewards.keys()]
        )


    def _set_obs(self):
        p = np.array(
            [np.sin((2 * np.pi * (self.phi))), np.cos((2 * np.pi * (self.phi)))]
        )

        sensor_data = self.data.sensordata
        self.obs = np.concatenate(
            [
                sensor_data,  # 0 - 28
                self.command,  # 29 - 31
                np.concatenate(
                    [
                        self.contact_force_left_foot[:3],
                        self.contact_force_right_foot[:3],
                    ]
                ),  # 32 - 37
                np.array([*p]),  # 38 - 39
            ]
        )

    def _get_obs(self):
        return self.obs


    def _normalize_quantity(self, name, q):
        k = 5.0
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
        return (q - self.exponents_ranges[name][0]) / (
            self.exponents_ranges[name][1] - self.exponents_ranges[name][0] + 1e-6
        )


    # computes the reward
    def _compute_reward(self, action: "npt.NDArray[np.float32]"):  # , symmetric_action):
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
            self.data.xpos[LEFT_FOOT, 2] < 0.05
            or self.data.xpos[RIGHT_FOOT, 2] < 0.05
            or np.linalg.norm(self.contact_force_left_foot) > 500
            or np.linalg.norm(self.contact_force_right_foot) > 500
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
        q_vz = np.exp(
            multiplicators["q_vz"]
            * np.linalg.norm(np.array([qvel[2]]) - np.array([self.z_cmd_vel])),
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
                        np.array(action).reshape(1, -1),
                        np.array(self.previous_action).reshape(1, -1),
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
        r_cmd = (+1.0 * q_vx + 0.8 * q_vy + 0.2 * q_orientation + 0.2 * q_vz) / (
            1.0 + 0.8 + 0.2 + 0.2
        )

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

        reward = self.normalize_reward(rewards, self.reward_coeffs)

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
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
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
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
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
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs,
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
