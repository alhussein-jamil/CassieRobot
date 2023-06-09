import logging as log
import os
from copy import deepcopy
from typing import Dict, Tuple

import gymnasium as gym
import mujoco as m
import numpy as np
import torch
import yaml
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import constants as c
import functions as f

log.basicConfig(level=log.DEBUG)

DEFAULT_CONFIG = {
    "steps_per_cycle": 30,
    "a_swing": 0,
    "b_swing": 0.5,
    "a_stance": 0.5,
    "b_stance": 1,
    "kappa": 25,
    "x_cmd_vel": 1.5,
    "y_cmd_vel": 0,
    "z_cmd_vel": 0,
    "terminate_when_unhealthy": True,
    "max_simulation_steps": 400,
    "pelvis_height": [0.6, 1.5],
    "feet_distance_x": [0.0, 1.0],
    "feet_distance_y": [0.0, 0.5],
    "feet_distance_z": [0.0, 0.5],
    "feet_pelvis_height": 0.3,
    "feet_height": 0.6,
    "model": "cassie",
    "render_mode": "rgb_array",
    "reset_noise_scale" : 0.01,
    "reward_coeffs": {
        "bias": 1.0,
        "r_biped": 4.0,
        "r_cmd" : 3.0,
        "r_smooth": 1.0,
        "r_alternate": 4.0

    }
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

    def __init__(self, env_config):
        DIRPATH = os.path.dirname(os.path.realpath(__file__))
        self._terminate_when_unhealthy = env_config.get("terminate_when_unhealthy", DEFAULT_CONFIG["terminate_when_unhealthy"])
        self._healthy_pelvis_z_range = env_config.get("pelvis_height", DEFAULT_CONFIG["pelvis_height"])
        self._healthy_feet_distance_x = env_config.get("feet_distance_x", DEFAULT_CONFIG["feet_distance_x"])
        self._healthy_feet_distance_y = env_config.get("feet_distance_y", DEFAULT_CONFIG["feet_distance_y"])
        self._healthy_feet_distance_z = env_config.get("feet_distance_z", DEFAULT_CONFIG["feet_distance_z"])
        self._healthy_dis_to_pelvis = env_config.get("feet_pelvis_height", DEFAULT_CONFIG["feet_pelvis_height"])
        self._healthy_feet_height = env_config.get("feet_height", DEFAULT_CONFIG["feet_height"])
        self._max_steps = env_config.get("max_simulation_steps", DEFAULT_CONFIG["max_simulation_steps"])
        self.steps_per_cycle = env_config.get("steps_per_cycle", DEFAULT_CONFIG["steps_per_cycle"])
        self.a_swing = env_config.get("a_swing", DEFAULT_CONFIG["a_swing"])
        self.a_stance = env_config.get("a_stance", DEFAULT_CONFIG["a_stance"])
        self.b_swing = env_config.get("b_swing", DEFAULT_CONFIG["b_swing"])
        self.b_stance = env_config.get("b_stance", DEFAULT_CONFIG["b_stance"])
        self.kappa = env_config.get("kappa", DEFAULT_CONFIG["kappa"])
        self.x_cmd_vel = env_config.get("x_cmd_vel", DEFAULT_CONFIG["x_cmd_vel"])
        self.y_cmd_vel = env_config.get("y_cmd_vel", DEFAULT_CONFIG["y_cmd_vel"])
        self.z_cmd_vel = env_config.get("z_cmd_vel", DEFAULT_CONFIG["z_cmd_vel"])
        self.model_file = env_config.get("model", "cassie")
        self.action_space = gym.spaces.Box(
            np.float32(c.low_action), np.float32(c.high_action)
        )
        self.reward_coeffs = env_config.get("reward_coeffs", DEFAULT_CONFIG["reward_coeffs"])
        self._reset_noise_scale = env_config.get("reset_noise_scale", DEFAULT_CONFIG["reset_noise_scale"])

        self.phi, self.steps, self.gamma_modified = 0, 0, 1
        self.previous_action = torch.zeros(10)
        self.gamma = env_config.get("gamma", 0.99)

        self.observation_space = Box(
            low=-1.2,
            high=1.2,
            shape=(25,),
        )

        MujocoEnv.__init__(
            self,
            model_path = DIRPATH + "/cassie-mujoco-sim-master/model/{}.xml".format(self.model_file),
            frame_skip = 20,
            render_mode=env_config.get("render_mode", None),
            observation_space=self.observation_space,
        )
        self.render_mode = "rgb_array"

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_pelvis_z_range

        self.isdone = "not done"

        if not min_z < self.data.xpos[c.PELVIS, 2] < max_z:
            self.isdone = "Pelvis not in range"

        if not self.steps <= self._max_steps:
            self.isdone = "Max steps reached"

        if (
            not self._healthy_feet_distance_x[0]
            < abs(self.data.xpos[c.LEFT_FOOT, 0] - self.data.xpos[c.RIGHT_FOOT, 0])
            < self._healthy_feet_distance_x[1]
        ):
            self.isdone = "Feet distance out of range along x-axis"

        if (
            not self._healthy_feet_distance_y[0]
            < abs(self.data.xpos[c.LEFT_FOOT, 1] - self.data.xpos[c.RIGHT_FOOT, 1])
            < self._healthy_feet_distance_y[1]
        ):
            self.isdone = "Feet distance out of range along y-axis"

        if (
            not self._healthy_feet_distance_z[0]
            < abs(self.data.xpos[c.LEFT_FOOT, 2] - self.data.xpos[c.RIGHT_FOOT, 2])
            < self._healthy_feet_distance_z[1]
        ):
            self.isdone = "Feet distance out of range along z-axis"

        if (
            self.contact
            and self.data.xpos[c.LEFT_FOOT, 2] >= self._healthy_feet_height
            and self.data.xpos[c.RIGHT_FOOT, 2] >= self._healthy_feet_height
        ):
            self.isdone = "Both Feet not on the ground"

        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[c.PELVIS, 2] - self.data.xpos[c.LEFT_FOOT, 2]
        ):
            self.isdone = "Left foot too close to pelvis"

        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[c.PELVIS, 2] - self.data.xpos[c.RIGHT_FOOT, 2]
        ):
            self.isdone = "Right foot too close to pelvis"

        if self.data.xpos[c.LEFT_FOOT, 1] > self.data.xpos[c.RIGHT_FOOT, 1]:
            self.isdone = "Feet Crossed"

        return self.isdone == "not done"

    @property
    def terminated(self):
        terminated = (
            (not self.is_healthy)
            if (self._terminate_when_unhealthy)  # or self.steps > c.MAX_STEPS)
            else False
        )
        return terminated

    def _get_obs(self):
        p = np.array(
            [np.sin((2 * np.pi * (self.phi))), np.cos((2 * np.pi * (self.phi)))]
        )
        temp = []

        # normalize the sensor data using sensor_ranges
        i = 0
        for key in c.sensor_ranges.keys():
            for x in self.data.sensor(key).data:
                temp.append(
                    (x - c.obs_ranges[0][i]) / (c.obs_ranges[1][i] - c.obs_ranges[0][i])
                )
                # temp.append(x)
                i += 1
        self.obs = np.clip(
            np.concatenate([temp, p]),
            self.observation_space.low,
            self.observation_space.high,
        )
        # getting the read positions of the sensors and concatenate the lists
        return self.obs

    def _get_symmetric_obs(self):
        obs = self._get_obs()
        symmetric_obs = deepcopy(obs)
        symmetric_obs[0:8] = obs[8:16]
        symmetric_obs[8:16] = obs[0:8]
        symmetric_obs[17] = -obs[17]
        symmetric_obs[19] = -obs[19]
        symmetric_obs[21] = -obs[21]
        symmetric_obs[24] = -obs[24]
        symmetric_obs[29] = obs[30]
        symmetric_obs[30] = obs[29]

        return symmetric_obs

    # computes the reward
    def compute_reward(self, action):
        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos = qpos[c.pos_index]
        qvel = qvel[c.vel_index]

        # Feet Contact Forces
        contacts = [contact.geom2 for contact in self.data.contact]
        contact_force_left_foot = np.zeros(6)
        contact_force_right_foot = np.zeros(6)
        if c.left_foot_force_idx in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(c.left_foot_force_idx),
                contact_force_left_foot,
            )
        if c.right_foot_force_idx in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(c.right_foot_force_idx),
                contact_force_right_foot,
            )

        # check if cassie hit the ground with feet
        if (
            self.data.xpos[c.LEFT_FOOT, 2] < 0.12
            or self.data.xpos[c.RIGHT_FOOT, 2] < 0.12
        ):
            self.contact = True

        # Some metrics to be used in the reward function
        q_vx = 1 - np.exp(
            c.multiplicators["q_vx"]
            * np.linalg.norm(np.array([qvel[0]]) - np.array([self.x_cmd_vel])) ** 2
        )
        q_vy = 1 - np.exp(
            c.multiplicators["q_vy"]
            * np.linalg.norm(np.array([qvel[1]]) - np.array([self.y_cmd_vel])) ** 2
        )
        q_vz = 1 - np.exp(
            c.multiplicators["q_vz"]
            * np.linalg.norm(np.array([qvel[2]]) - np.array([self.z_cmd_vel])) ** 2
        )

        q_left_frc = 1.0 - np.exp(
            c.multiplicators["q_frc"] * np.linalg.norm(contact_force_left_foot) ** 2
        )
        q_right_frc = 1.0 - np.exp(
            c.multiplicators["q_frc"] * np.linalg.norm(contact_force_right_foot) ** 2
        )
        q_left_spd = 1.0 - np.exp(
            c.multiplicators["q_spd"] * np.linalg.norm(qvel[12]) ** 2
        )
        q_right_spd = 1.0 - np.exp(
            c.multiplicators["q_spd"] * np.linalg.norm(qvel[19]) ** 2
        )
        q_action_diff = 1 - np.exp(
            c.multiplicators["q_action"]
            * float(
                f.action_dist(
                    torch.tensor(action).reshape(1, -1),
                    torch.tensor(self.previous_action).reshape(1, -1),
                )
            )
        )
        q_orientation = 1 - np.exp(
            c.multiplicators["q_orientation"]
            * (
                1
                - (
                    (self.data.sensor("pelvis-orientation").data.T)
                    @ (c.FORWARD_QUARTERNIONS)
                )
                ** 2
            )
        )
        q_torque = 1 - np.exp(c.multiplicators["q_torque"] * np.linalg.norm(action))
        q_pelvis_acc = 1 - np.exp(
            c.multiplicators["q_pelvis_acc"]
            * (np.linalg.norm(self.data.sensor("pelvis-angular-velocity").data))
        )

        cycle_steps = float(self.steps % self.steps_per_cycle) / self.steps_per_cycle
        phase_left = (
            1 if 0.75 > cycle_steps >= 0.5 else -1 if 1 > cycle_steps >= 0.75 else 0
        )
        phase_right = (
            1 if 0.25 > cycle_steps >= 0 else -1 if 0.5 > cycle_steps >= 0.25 else 0
        )

        q_phase_left = 1.0 - np.exp(
            c.multiplicators["q_marche_distance"]
            * np.clip(
                phase_left
                * (
                    0.2
                    + self.data.xpos[c.LEFT_FOOT][0]
                    - self.data.xpos[c.RIGHT_FOOT][0]
                ),
                0,
                np.inf,
            )
        )
        q_phase_right = 1.0 - np.exp(
            c.multiplicators["q_marche_distance"]
            * np.clip(
                phase_right
                * (
                    0.2
                    + self.data.xpos[c.RIGHT_FOOT][0]
                    - self.data.xpos[c.LEFT_FOOT][0]
                ),
                0,
                np.inf,
            )
        )

        q_feet_orientation_left = 1 - np.exp(
            c.multiplicators["q_feet_orientation"]
            * np.abs(
                self.data.sensordata[c.LEFT_FOOT_JOINT] - c.target_feet_orientation
            )
        )
        q_feet_orientation_right = 1 - np.exp(
            c.multiplicators["q_feet_orientation"]
            * np.abs(
                self.data.sensordata[c.RIGHT_FOOT_JOINT] - c.target_feet_orientation
            )
        )

        self.exponents = {
            "q_vx": np.linalg.norm(np.array([qvel[0]]) - np.array([self.x_cmd_vel]))
            ** 2,
            "q_vy": np.linalg.norm(np.array([qvel[1]]) - np.array([self.y_cmd_vel]))
            ** 2,
            "q_vz": np.linalg.norm(np.array([qvel[2]]) - np.array([self.z_cmd_vel]))
            ** 2,
            "q_left_frc": np.linalg.norm(contact_force_left_foot) ** 2,
            "q_right_frc": np.linalg.norm(contact_force_right_foot) ** 2,
            "q_left_spd": np.linalg.norm(qvel[12]) ** 2,
            "q_right_spd": np.linalg.norm(qvel[19]) ** 2,
            "q_action_diff": float(
                f.action_dist(
                    torch.tensor(action).reshape(1, -1),
                    torch.tensor(self.previous_action).reshape(1, -1),
                )
            ),
            "q_orientation": (
                1
                - (
                    (self.data.sensor("pelvis-orientation").data.T)
                    @ (c.FORWARD_QUARTERNIONS)
                )
                ** 2
            ),
            "q_torque": np.linalg.norm(action),
            "q_pelvis_acc": np.linalg.norm(
                self.data.sensor("pelvis-angular-velocity").data
            )
            + np.linalg.norm(
                self.data.sensor("pelvis-linear-acceleration").data
                - self.model.opt.gravity.data
            ),
            "feet distance": np.linalg.norm(
                self.data.xpos[c.LEFT_FOOT][0] - self.data.xpos[c.RIGHT_FOOT][0]
            ),
            "pelvis_height ": self.data.xpos[c.PELVIS, 2],
            "oriantation_right_foot": np.abs(
                self.data.sensordata[c.RIGHT_FOOT_JOINT] - c.target_feet_orientation
            ),
            "oriantation_left_foot": np.abs(
                self.data.sensordata[c.LEFT_FOOT_JOINT] - c.target_feet_orientation
            ),
        }
        used_quantities = {
            "q_vx": q_vx,
            "q_vy": q_vy,
            "q_vz": q_vz,
            "q_left_frc": q_left_frc,
            "q_right_frc": q_right_frc,
            "q_left_spd": q_left_spd,
            "q_right_spd": q_right_spd,
            "q_action_diff": q_action_diff,
            "q_orientation": q_orientation,
            "q_torque": q_torque,
            "q_pelvis_acc": q_pelvis_acc,
            "q_phase_left": q_phase_left,
            "q_phase_right": q_phase_right,
            "q_feet_orientation_left": q_feet_orientation_left,
            "q_feet_orientation_right": q_feet_orientation_right,
        }

        self.used_quantities = used_quantities

        # Responsable for the swing and stance phase
        def i(phi, a, b):
            return f.p_between_von_mises(a = a , b = b, kappa = 25 , x = phi)
            return f.von_mises_approx(a, b, self.kappa, phi)

        def i_swing_frc(phi):
            return i(phi, self.a_swing, self.b_swing)

        def i_swing_spd(phi):
            return i(phi, self.a_swing, self.b_swing)

        def i_stance_spd(phi):
            return i(phi, self.a_stance, self.b_stance)

        def i_stance_frc(phi):
            return i(phi, self.a_stance, self.b_stance)

        def c_frc(phi):
            return c.c_swing_frc * i_swing_frc(phi) + c.c_stance_frc * i_stance_frc(phi)

        def c_spd(phi):
            return c.c_swing_spd * i_swing_spd(phi) + c.c_stance_spd * i_stance_spd(phi)

        r_alternate = (-1.0 * q_phase_left - 1.0 * q_phase_right) / (1.0 + 1.0)

        r_cmd = (
            -1.0 * q_vx
            - 1.0 * q_vy
            - 1.0 * q_orientation
            - 0.5 * q_vz
            - 1.0 * q_feet_orientation_left
            - 1.0 * q_feet_orientation_right
        ) / (1.0 + 1.0 + 1.0 + 0.5 + 1.0 + 1.0)

        r_smooth = (-1.0 * q_action_diff - 1.0 * q_torque - 1.0 * q_pelvis_acc) / (
            1.0 + 1.0 + 1.0
        )

        r_biped = 0
        r_biped += c_frc(self.phi + c.THETA_LEFT) * q_left_frc
        r_biped += c_frc(self.phi + c.THETA_RIGHT) * q_right_frc
        r_biped += c_spd(self.phi + c.THETA_LEFT) * q_left_spd
        r_biped += c_spd(self.phi + c.THETA_RIGHT) * q_right_spd

        r_biped /= 2.0


        rewards = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
            "r_alternate": r_alternate,
        }
        reward = self.reward_coeffs["bias"] + sum([rewards[k]*self.reward_coeffs[k] for k in rewards.keys()])/sum(self.reward_coeffs.values())
    
        self.C = {
            "sin": self.obs[-2],
            "cos": self.obs[-1],
            "phase_right": phase_right,
            "phase_left": phase_left,
            "C_frc_left": c_frc(self.phi + c.THETA_LEFT),
            "C_frc_right": c_frc(self.phi + c.THETA_RIGHT),
            "C_spd_left": c_spd(self.phi + c.THETA_LEFT),
            "C_spd_right": c_spd(self.phi + c.THETA_RIGHT),
        }

        metrics = {
            "dis_x": self.data.geom_xpos[16, 0],
            "dis_y": self.data.geom_xpos[16, 1],
            "dis_z": self.data.geom_xpos[16, 2],
            "vel_x": self.data.qvel[0],
            "vel_y": self.data.qvel[1],
            "vel_z": self.data.qvel[2],
            "feet_distance_x": abs(
                self.data.xpos[c.LEFT_FOOT, 0] - self.data.xpos[c.RIGHT_FOOT, 0]
            ),
            "feet_distance_y": abs(
                self.data.xpos[c.LEFT_FOOT, 1] - self.data.xpos[c.RIGHT_FOOT, 1]
            ),
            "feet_distance_z": abs(
                self.data.xpos[c.LEFT_FOOT, 2] - self.data.xpos[c.RIGHT_FOOT, 2]
            ),
        }
        return reward, used_quantities, rewards, metrics

    # step in time
    def step(self, action):
        # clip the action to the ranges in action_space (done inside the config
        # that's why removed)
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        m.mj_step(self.model, self.data)

        observation = self._get_obs()

        reward, used_quantities, rewards, metrics = self.compute_reward(action)

        terminated = self.terminated

        self.steps += 1
        self.phi += 1.0 / self.steps_per_cycle
        self.phi = self.phi % 1

        self.previous_action = action

        self.gamma_modified *= self.gamma
        info = {}
        info["custom_rewards"] = rewards
        info["custom_quantities"] = used_quantities

        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        info["other_metrics"] = metrics

        return observation, reward, terminated, False, info

    # resets the simulation
    def reset_model(self, seed=0):
        m.mj_inverse(self.model, self.data)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.previous_action = np.zeros(10)
        self.phi = np.random.randint(0, self.steps_per_cycle) / self.steps_per_cycle
        self.phi = 0
        self.steps = 0
        self.contact = False
        # self.rewards = {"R_biped": 0, "R_cmd": 0, "R_smooth": 0}

        self.gamma_modified = 1
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation


class MyCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
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
        **kwargs
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
        **kwargs
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
