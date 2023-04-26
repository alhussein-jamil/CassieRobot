
from typing import Dict, Tuple
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
import numpy as np
import gymnasium as gym
import gymnasium.utils as utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import constants as c
import functions as f
import mujoco as m
import torch
import logging as log
import os 

log.basicConfig(level=log.DEBUG)


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, config, **kwargs):
        DIRPATH = os.path.dirname(os.path.realpath(__file__))
        utils.EzPickle.__init__(self, config, **kwargs)
        self._terminate_when_unhealthy = config.get(
            "terminate_when_unhealthy", True)
        self._healthy_z_range = config.get("healthy_z_range", (0.35, 2.0))


        self.action_space = gym.spaces.Box(
            np.float32(c.low_action), np.float32(c.high_action)
        )

        self._reset_noise_scale = config.get("reset_noise_scale", 1e-2)
        self.phi, self.steps, self.gamma_modified = 0, 0, 1
        self.previous_action = torch.zeros(10)
        self.gamma = config.get("gamma", 0.99)
        
        self.observation_space = Box(
            low=np.float32(
                np.array(c.low_obs)), high=np.float32(
                np.array(c.high_obs)), shape=(
                25,))

        MujocoEnv.__init__(
            self,
            config.get(
                "model_path",
                DIRPATH + "/cassie-mujoco-sim-master/model/cassie.xml",
            ),
            20,
            render_mode=config.get(
                "render_mode",
                None),
            observation_space=self.observation_space,
            **kwargs)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        # it is healthy if in range and one of the feet is on the ground
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (
            (not self.is_healthy)
            if (self._terminate_when_unhealthy or self.steps > c.MAX_STEPS)
            else False
        )
        return terminated

    def _get_obs(self):
        p = np.array([np.sin((2 * np.pi * (self.phi))),
                      np.cos((2 * np.pi * (self.phi)))])
        temp = []
        # normalize the sensor data using sensor_ranges
        # self.data.sensor('pelvis-orientation').data
        for key in c.sensor_ranges.keys():
            temp.append(f.normalize(key, self.data.sensor(key).data))
        temp = np.array(np.concatenate(temp))

        # getting the read positions of the sensors and concatenate the lists
        return np.concatenate([temp, p])

    # computes the reward
    def compute_reward(self, action):
        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()



        qpos = qpos[c.pos_index]
        qvel = qvel[c.vel_index]

        # Feet Contact Forces
        contact_force_right_foot = np.zeros(6)
        m.mj_contactForce(self.model, self.data, 0, contact_force_right_foot)
        contact_force_left_foot = np.zeros(6)
        m.mj_contactForce(self.model, self.data, 1, contact_force_left_foot)

        # Some metrics to be used in the reward function
        q_vx = 1 - np.exp(
            c.multiplicators["q_vx"]
            * np.linalg.norm(np.array([qvel[0]]) - np.array([c.X_VEL])) ** 2
        )
        q_vy = 1 - np.exp(
            c.multiplicators["q_vy"]
            * np.linalg.norm(np.array([qvel[1]]) - np.array([c.Y_VEL])) ** 2
        )
        q_vz = 1 - np.exp(
            c.multiplicators["q_vz"]
            * np.linalg.norm(np.array([qvel[2]]) - np.array([c.Z_VEL])) ** 2
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
        q_torque = 1 - \
            np.exp(c.multiplicators["q_torque"] * np.linalg.norm(action))
        q_pelvis_acc = 1 - np.exp(
            c.multiplicators["q_pelvis_acc"]
            * (np.linalg.norm(self.data.sensor("pelvis-angular-velocity").data))
        )  

        self.exponents = {
            "q_vx": np.linalg.norm(np.array([qvel[0]]) - np.array([c.X_VEL])) ** 2,
            "q_vy": np.linalg.norm(np.array([qvel[1]]) - np.array([c.Y_VEL])) ** 2,
            "q_vz": np.linalg.norm(np.array([qvel[2]]) - np.array([c.Z_VEL])) ** 2,
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
        }
        
        
        
        self.used_quantities = used_quantities
        # Responsable for the swing and stance phase
        def i(phi, a, b): return f.p_between_von_mises(a, b, c.KAPPA, phi)
        def i_swing_frc(phi): 
            return i(phi, c.a_swing, c.b_swing)
        def i_swing_spd(phi): 
            return i(phi, c.a_swing, c.b_swing)
        def i_stance_spd(phi): 
            return i(phi, c.a_stance, c.b_stance)
        def i_stance_frc(phi): 
            return i(phi, c.a_stance, c.b_stance)

        def c_frc(phi): 
            return c.c_swing_frc * i_swing_frc(
            phi
        ) + c.c_stance_frc * i_stance_frc(phi)

        def c_spd(phi): 
            return c.c_swing_spd * i_swing_spd(
            phi
        ) + c.c_stance_spd * i_stance_spd(phi)

        r_cmd = -1.0 * q_vx - 1.0 * q_vy - 1.0 * q_orientation - 0.5 * q_vz
        r_smooth = -1.0 * q_action_diff - 1.0 * q_torque - 1.0 * q_pelvis_acc
        
        r_biped = 0
        r_biped += c_frc(self.phi + c.THETA_LEFT) * q_left_frc
        r_biped += c_frc(self.phi + c.THETA_RIGHT) * q_right_frc
        r_biped += c_spd(self.phi + c.THETA_LEFT) * q_left_spd
        r_biped += c_spd(self.phi + c.THETA_RIGHT) * q_right_spd

        reward = 2.5 + 0.5 * r_biped + 0.375 * r_cmd + 0.125 * r_smooth

        rewards = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth}
        self.C = {"C_frc_left": c_frc(self.phi + c.THETA_LEFT),
                  "C_frc_right": c_frc(self.phi + c.THETA_RIGHT),
                  "C_spd_left": c_spd(self.phi + c.THETA_LEFT),
                  "C_spd_right": c_spd(self.phi + c.THETA_RIGHT)}
        return reward,used_quantities,rewards

    # step in time
    def step(self, action):
        # clip the action to the ranges in action_space (done inside the config
        # that's why removed)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        reward,used_quantities, rewards= self.compute_reward(action)

        terminated = self.terminated

        self.steps += 1
        self.phi += 1.0 / c.STEPS_IN_CYCLE
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
        return observation, reward, terminated, False, info

    # resets the simulation
    def reset_model(self):
        m.mj_inverse(self.model, self.data)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.previous_action = np.zeros(10)
        self.phi = 0
        self.steps = 0
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
