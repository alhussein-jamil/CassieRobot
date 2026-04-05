from pathlib import Path

import mediapy as media
from gymnasium import Env
from loguru import logger
from ray.rllib.algorithms.ppo import PPO


class Evaluator:
    """Handles model evaluation and video rendering."""

    def __init__(self, run_config: dict):
        self.render_fps = run_config.get("render_fps", 30)

    def evaluate(
        self,
        trainer: PPO,
        env: Env,
        epoch: int,
        config_index: int,
        test_dir: Path,
        sim_fps: int,
    ) -> None:
        """Run one evaluation episode and save a video."""
        env.reset()
        obs, _ = env.reset()
        done = False
        frames = []
        steps = 0
        render_step = 0

        policy = trainer.get_policy()

        render_fps = self.render_fps
        if render_fps <= 0 or sim_fps <= 0:
            logger.warning("render_fps or sim_fps is non-positive. Skipping rendering.")
            render_fps = 0

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]

            obs, _, done, _, _ = env.step(action)

            should_render = False
            if render_fps > 0 and steps * render_fps >= render_step * sim_fps:
                should_render = True
                render_step += 1

            if should_render:
                frame = env.render()
                frames.append(frame)

            steps += 1

        logger.info("Episode finished after {} steps", steps)
        logger.info("Episode done reason: {}", env.isdone)

        video_path = test_dir / f"config_{config_index}/run_{epoch}.mp4"
        media.write_video(video_path, frames, fps=min(render_fps, sim_fps))
        logger.info("Test saved at {}", video_path)
