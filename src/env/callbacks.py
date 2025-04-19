import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.callbacks.callbacks import RLlibCallback


class CassieEnvCallback(RLlibCallback):
    def on_episode_step(self, *, episode: SingleAgentEpisode, env, **kwargs):
        """Store reward components at each step in the episode."""
        # Get the info dict from the last step
        if len(episode.infos) > 0:
            info = episode.infos[-1]

            # Store reward components from custom_metrics if available
            if "custom_metrics" in info:
                # Automatically detect any reward component (keys starting with 'r_')
                for key, value in info["custom_metrics"].items():
                    if key.startswith("r_"):
                        episode.add_temporary_timestep_data(key, value)

    def on_episode_end(self, *, episode: SingleAgentEpisode, metrics_logger, **kwargs):
        """Log aggregated reward components at the end of episode."""
        # Get all reward component keys from temporary data
        all_keys = set()
        for i in range(len(episode)):
            if len(episode.infos) > i:
                info = episode.infos[i]
                if "custom_metrics" in info:
                    # Add any reward component key
                    all_keys.update(
                        key for key in info["custom_metrics"] if key.startswith("r_")
                    )

        # Log each discovered reward component
        for component in all_keys:
            values = episode.get_temporary_timestep_data(component)
            if values:
                # Log mean, min, max for each component
                metrics_logger.log_value(
                    f"{component}_mean", np.mean(values), reduce="mean", window=50
                )
                metrics_logger.log_value(
                    f"{component}_min", np.min(values), reduce="min", window=50
                )
                metrics_logger.log_value(
                    f"{component}_max", np.max(values), reduce="max", window=50
                )

        # Log essential environment metrics
        if len(episode.infos) > 0 and "custom_metrics" in episode.infos[-1]:
            custom_metrics = episode.infos[-1]["custom_metrics"]
            for metric_name, metric_value in custom_metrics.items():
                # Only log non-reward metrics (distance, height)
                if not metric_name.startswith("r_"):
                    metrics_logger.log_value(
                        f"env_{metric_name}", metric_value, reduce="mean", window=50
                    )
