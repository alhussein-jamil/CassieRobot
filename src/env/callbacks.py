from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2 as Episode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
import numpy as np

class MyCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        episode: Episode | SingleAgentEpisode,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        if isinstance(episode, Episode):
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
        episode: Episode,
        **kwargs,
    ):
        if isinstance(episode, Episode):
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
        if isinstance(episode, Episode):
            if "num_batches" not in episode.custom_metrics:
                episode.custom_metrics["num_batches"] = 0
            episode.custom_metrics["num_batches"] += 1
