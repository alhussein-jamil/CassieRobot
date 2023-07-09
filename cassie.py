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
    "max_yaw": 0.2
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
        self.action_space = gym.spaces.Box(
            np.float32(c.low_action), np.float32(c.high_action)
        )
        phis = np.linspace(0, 1, self.steps_per_cycle, endpoint=False)
        self.von_mises_values_swing = np.array([
            f.p_between_von_mises(a=self.a_swing, b=self.b_swing, kappa=self.kappa, x=p)
            for p in phis
        ]) * 2.0 - 1.0
        self.von_mises_values_stance =  np.array([
            f.p_between_von_mises(
                a=self.a_stance, b=self.b_stance, kappa=self.kappa, x=p
            )
            for p in phis
        ] ) * 2.0 - 1.0

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

        self.observation_space = Box(
            low= - np.inf,
            high= np.inf,
            shape=(31,),
        )
        self.reward_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
        )
        self.metadata["render_fps"] = 2000 // env_config.get("frame_skip", 40)
        MujocoEnv.__init__(
            self,
            model_path=DIRPATH
            + "/cassie-mujoco-sim-master/model/{}.xml".format(self.model_file),
            frame_skip=env_config.get("frame_skip", 40),
            render_mode=env_config.get("render_mode", None),
            observation_space=self.observation_space,
        )
        # self.action_space = Box(
        #     low=np.array([*self.action_space.low, *self.action_space.low]),
        #     high=np.array([*self.action_space.high, *self.action_space.high]),
        # )
        self.render_mode = "rgb_array"
        # if self.training:
        #     self.symmetric_turn = True
        # else:
        #     self.symmetric_turn = False
        self.symmetric_turn = np.random.choice([True, False])
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )


    def quat_to_rpy(self,quaternion):
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
        if self.contact and not min_z < self.data.xpos[c.PELVIS, 2] < max_z:
            self.isdone = "Pelvis not in range"
            self.done_n = 1.0
        if not self.steps <= self._max_steps:
            self.isdone = "Max steps reached"
            self.done_n = 2.0
        if (
            not abs(self.data.xpos[c.LEFT_FOOT, 0] - self.data.xpos[c.RIGHT_FOOT, 0])
            < self._healthy_feet_distance_x
        ):
            self.isdone = "Feet distance out of range along x-axis"
            self.done_n = 3.0
        if (
            not 0.0
            < self.data.xpos[c.RIGHT_FOOT, 1] - self.data.xpos[c.LEFT_FOOT, 1]
            < self._healthy_feet_distance_y
        ):
            self.isdone = "Feet distance out of range along y-axis"

        if (
            not abs(self.data.xpos[c.LEFT_FOOT, 2] - self.data.xpos[c.RIGHT_FOOT, 2])
            < self._healthy_feet_distance_z
        ):
            self.isdone = "Feet distance out of range along z-axis"
            self.done_n = 4.0
        if (
            self.contact
            and self.data.xpos[c.LEFT_FOOT, 2] >= self._healthy_feet_height
            and self.data.xpos[c.RIGHT_FOOT, 2] >= self._healthy_feet_height
            and np.linalg.norm(self.contact_force_left_foot) < 0.01
            and np.linalg.norm(self.contact_force_right_foot) < 0.01
        ):
            self.isdone = "Both Feet not on the ground"
            self.done_n = 5.0
        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[c.PELVIS, 2] - self.data.xpos[c.LEFT_FOOT, 2]
        ):
            self.isdone = "Left foot too close to pelvis"
            self.done_n = 6.0
        if (
            not self._healthy_dis_to_pelvis
            < self.data.xpos[c.PELVIS, 2] - self.data.xpos[c.RIGHT_FOOT, 2]
        ):
            self.isdone = "Right foot too close to pelvis"
            self.done_n = 7.0
        pelvis_rpy = self.quat_to_rpy(self.data.sensordata[16:20])
        rpy_diff = np.abs(pelvis_rpy - self.init_rpy)

        if ( 
            rpy_diff[0] > self.max_roll
        ):
            self.isdone = "Roll too high"
            self.done_n = 8.0
        if (
            rpy_diff[1] > self.max_pitch
        ):
            self.isdone = "Pitch too high"
            self.done_n = 9.0
        if (
            rpy_diff[2] > self.max_yaw
        ):
            self.isdone = "Yaw too high"
            self.done_n = 10.0

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

        sensor_data = self.data.sensordata

        self.dict_obs = dict(
            actuator_pos = np.concatenate([sensor_data[:5], sensor_data[8:13]]),
            joint_pos = np.concatenate([sensor_data[5:8], sensor_data[13:16]]),
            pelvis_orientation = sensor_data[16:20],
            pelvis_angular_velocity = sensor_data[20:23],
            pelvis_linear_acceleration = sensor_data[23:26],
            magnetometer = sensor_data[26:29],
        )

        # sensor_data = 2.0 * (sensor_data - c.sensor_ranges[0, :]) / ( c.sensor_ranges[1,:] - c.sensor_ranges[0,:]) - 1.0

        # getting the read positions of the sensors and concatenate the lists
        self.obs = np.array([*sensor_data, *p])

        return self.obs

    def _get_symmetric_obs(self):
        obs = self._get_obs()

        symmetric_obs = deepcopy(obs)

        symmetric_obs[0:8] = obs[8:16]
        symmetric_obs[8:16] = obs[0:8]

        # pelvis quaternion symmetric along xoz
        symmetric_obs[17] = -obs[17]
        symmetric_obs[18] = -obs[18]

        # pelvis angular velocity symmetric along xoz
        symmetric_obs[20] = -obs[20]
        symmetric_obs[22] = -obs[22]

        # pelvis acceleration symmetric along xoz
        symmetric_obs[24] = -obs[24]
        

        # symmetry of the clock
        symmetric_obs[29] = - obs[ 29 ]
        symmetric_obs[30] = - obs[ 30 ]

        return symmetric_obs

    def symmetric_action(self, action):
        return np.array([*action[5:], *action[:5]])

    # computes the reward
    def compute_reward(self, action):#, symmetric_action):
        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos = qpos[c.pos_index]
        qvel = qvel[c.vel_index]

        # Feet Contact Forces
        contacts = [contact.geom2 for contact in self.data.contact]
        self. contact_force_left_foot = np.zeros(6)

        self.contact_force_right_foot = np.zeros(6)
        if c.left_foot_force_idx in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(c.left_foot_force_idx),
                self.contact_force_left_foot,
            )
        if c.right_foot_force_idx in contacts:
            m.mj_contactForce(
                self.model,
                self.data,
                contacts.index(c.right_foot_force_idx),
                self.contact_force_right_foot,
            )

        # check if cassie hit the ground with feet
        if (
            # self.data.xpos[c.LEFT_FOOT, 2] < 0.12
            # or self.data.xpos[c.RIGHT_FOOT, 2] < 0.12

            np.linalg.norm(self.contact_force_left_foot) > 10
            or np.linalg.norm(self.contact_force_right_foot) > 10
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
            c.multiplicators["q_frc"] * np.linalg.norm(self.contact_force_left_foot) ** 2
        )
        q_right_frc = 1.0 - np.exp(
            c.multiplicators["q_frc"] * np.linalg.norm(self.contact_force_right_foot) ** 2
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
                    np.array(action).reshape(1, -1),
                    np.array(self.previous_action).reshape(1, -1),
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
        # q_symmetric = 1 - np.exp(
        #     c.multiplicators["q_symmetric"]
        #     * float(
        #         f.action_dist(
        #             np.array(action).reshape(1, -1),
        #             np.array(symmetric_action).reshape(1, -1),
        #         )
        #     )
        # )
        q_symmetric = 0.0

        def i_swing_frc(phi):
            index = round(phi * self.steps_per_cycle) % self.steps_per_cycle
            return self.von_mises_values_swing[index]

        def i_stance_spd(phi):
            index = round(phi * self.steps_per_cycle) % self.steps_per_cycle
            return self.von_mises_values_stance[index]

        def c_frc(phi):
            return c.c_swing_frc * i_swing_frc(phi)

        def c_spd(phi):
            return c.c_stance_spd * i_stance_spd(phi)

        r_alternate = (-1.0 * q_phase_left - 1.0 * q_phase_right) / (1.0 + 1.0)

        r_cmd = (
            -1.0 * q_vx
            - 1.0 * q_vy
            - 1.0 * q_orientation
            - 0.5 * q_vz
            - 1.0 * q_feet_orientation_left
            - 1.0 * q_feet_orientation_right
        ) / (1.0 + 1.0 + 1.0 + 0.5 + 1.0 + 1.0)

        r_smooth = (-0.1 * q_action_diff - 1.0 * q_torque - 0.1 * q_pelvis_acc) / (
            0.1 + 1.0 + 0.1
        )

        r_biped = 0
        r_biped += c_frc(self.phi + c.THETA_LEFT) * q_left_frc
        r_biped += c_frc(self.phi + c.THETA_RIGHT) * q_right_frc
        r_biped += c_spd(self.phi + c.THETA_LEFT) * q_left_spd
        r_biped += c_spd(self.phi + c.THETA_RIGHT) * q_right_spd
        r_biped -= 2.0
        r_biped /= 4.0

        r_symmetric = -1.0 * q_symmetric

        rewards = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
            "r_alternate": r_alternate,
            "r_symmetric": r_symmetric,
        }

        reward = self.reward_coeffs["bias"] + sum(
            [rewards[k] * self.reward_coeffs[k] for k in rewards.keys()]
        ) / sum(self.reward_coeffs.values())

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

        self.exponents = {
            "q_vx": np.linalg.norm(np.array([qvel[0]]) - np.array([self.x_cmd_vel]))
            ** 2,
            "q_vy": np.linalg.norm(np.array([qvel[1]]) - np.array([self.y_cmd_vel]))
            ** 2,
            "q_vz": np.linalg.norm(np.array([qvel[2]]) - np.array([self.z_cmd_vel]))
            ** 2,
            "q_frc_left": np.linalg.norm(self.contact_force_left_foot) ** 2,
            "q_frc_right": np.linalg.norm(self.contact_force_right_foot) ** 2,
            "q_spd_left": np.linalg.norm(qvel[12]) ** 2,
            "q_spd_right": np.linalg.norm(qvel[19]) ** 2,
            "q_action": float(
                f.action_dist(
                    np.array(action).reshape(1, -1),
                    np.array(self.previous_action).reshape(1, -1),
                )
            ),
            "q_orientation_left": (
                np.abs(
                    self.data.sensordata[c.LEFT_FOOT_JOINT] - c.target_feet_orientation
                )
            ),
            "q_orientation_right": (
                np.abs(
                    self.data.sensordata[c.RIGHT_FOOT_JOINT] - c.target_feet_orientation
                )
            ),
            "q_torque": np.linalg.norm(action),
            "q_pelvis_acc": np.linalg.norm(
                self.data.sensor("pelvis-angular-velocity").data
            ),
            # "q_symmetric": float(
            #     f.action_dist(
            #         np.array(action).reshape(1, -1),
            #         np.array(symmetric_action).reshape(1, -1),
            #     )
            # ),
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
        return reward, rewards, metrics

    # step in time
    def step(self, action):

        if self.symmetric_turn:
            act = self.symmetric_action(action)
            # act, sym_act = action[len(action) // 2 :], self.symmetric_action(
            #     action[: len(action) // 2]
            # )
        else:
            act = action
            # act, sym_act = action[: len(action) // 2], self.symmetric_action(
            #     action[len(action) // 2 :]
            # )

        self.do_simulation(act, self.frame_skip)

        m.mj_step(self.model, self.data)
        terminated = self.terminated
        reward, rewards, metrics = self.compute_reward(act)#, sym_act)
        
        # if self.training:
        # self.symmetric_turn = np.random.choice([True, False])

        if self.symmetric_turn:
            observation = self._get_symmetric_obs()
        else:
            observation = self._get_obs()


        self.steps += 1
        self.phi += 1.0 / self.steps_per_cycle
        self.phi = self.phi % 1

        self.previous_action = act
        # self.previous_symmetric_action = sym_act

        info = {}
        info["custom_rewards"] = rewards
        # info["custom_quantities"] = used_quantities

        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        info["other_metrics"] = metrics

        return observation, reward, terminated, False, info
        # return observation, torch.tensor(list(rewards.values())), terminated, False, info

    # resets the simulation
    def reset_model(self, seed=0):
        m.mj_inverse(self.model, self.data)

        noise_low = -self._reset_noise_scale
        self.done_n = 0.0
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10)

        self.phi = np.random.randint(0, self.steps_per_cycle) / self.steps_per_cycle

        self.steps = 0

        self.contact = False
        # if self.training:
        #     self.symmetric_turn = True
        # else: 
        #     self.symmetric_turn = False
        self.symmetric_turn =not self.symmetric_turn
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        self.init_rpy =  self.quat_to_rpy(self.data.sensordata[16:20])

        return self._get_symmetric_obs() if self.symmetric_turn else self._get_obs()

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
