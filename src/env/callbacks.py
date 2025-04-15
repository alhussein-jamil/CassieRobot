import math
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.callbacks.callbacks import RLlibCallback

from gymnasium.vector import VectorEnv


class CassieEnvCallback(RLlibCallback):
    pass
    # def on_episode_step(self, *, episode: SingleAgentEpisode, env: VectorEnv, **kwargs):
    #     pass

    # def on_episode_end(self, *, episode: SingleAgentEpisode, metrics_logger, **kwargs):
    #     theta1s = episode.get_temporary_timestep_data("theta1")
    #     avg_theta1 = np.mean(theta1s)

    #     # Log the resulting average angle - per episode - to the MetricsLogger.
    #     # Report with a sliding window of 50.
    #     metrics_logger.log_value("test", avg_theta1, reduce="mean", window=50)
