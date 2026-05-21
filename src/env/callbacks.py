import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CassieEnvCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """Store reward components at each step in the episode."""
        if not hasattr(episode, "user_data"):
            return

        info = episode.last_info_for()
        if info and "custom_metrics" in info:
            for key, value in info["custom_metrics"].items():
                if key.startswith("r_"):
                    if key not in episode.user_data:
                        episode.user_data[key] = []
                    episode.user_data[key].append(value)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        """Log aggregated reward components at the end of episode."""
        if not hasattr(episode, "user_data"):
            return

        for key, values in episode.user_data.items():
            if key.startswith("r_") and values:
                episode.custom_metrics[f"{key}_mean"] = np.mean(values)
                episode.custom_metrics[f"{key}_min"] = np.min(values)
                episode.custom_metrics[f"{key}_max"] = np.max(values)

        info = episode.last_info_for()
        if info and "custom_metrics" in info:
            for metric_name, metric_value in info["custom_metrics"].items():
                if not metric_name.startswith("r_"):
                    episode.custom_metrics[f"env_{metric_name}"] = metric_value
