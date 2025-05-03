import argparse
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mediapy as media
import numpy as np
import ray
import torch
import yaml
from gymnasium import Env
from loguru import logger
from pyswarms.single.global_best import GlobalBestPSO
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from src.env.callbacks import CassieEnvCallback
from src.env.cassie import CassieEnv
from src.training.utils import fill_dict_with_list, flatten_dict
from src.training.loader import Loader

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger.remove()
logger.add(
    lambda msg: print(msg), colorize=True, format="<level>{level}</level> | {message}"
)

# Check for CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    output_dir: str = "output"
    checkpoint_frequency: int = 300  # seconds
    simulation_frequency: int = 600  # seconds
    clean_run: bool = False
    config_path: str = "configs/default_config.yaml"
    use_swarm: bool = False
    max_test_i: int = 0


class TrainingManager:
    """Manages the training and evaluation process for Cassie robot."""

    def __init__(self, config: TrainingConfig):
        """Initialize training manager with configuration."""
        self.config = config
        self.last_checkpoint_time = time.time() - np.inf
        self.last_render_time = time.time() - np.inf

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.sim_dir = self.output_dir / "simulations"
        self.sim_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        self.log_dir = self.output_dir / "ray_results"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Find latest test directory
        self.find_latest_test_dir()

        # Initialize ray
        ray.init(ignore_reinit_error=True, num_gpus=1)
        register_env("cassie-v0", lambda cfg: CassieEnv(cfg))

        # Load configuration
        self.loader = Loader(logdir=self.log_dir, simdir=self.sim_dir)
        self.full_config = self.loader.load_config(config.config_path)

        # set the number of env runners to the number of GPUs
        self.full_config["training"]["env_runners"]["num_env_runners"] = (
            multiprocessing.cpu_count()
        )
        self.full_config["training"]["env_runners"]["num_gpus_per_env_runner"] = (
            1.0 / multiprocessing.cpu_count()
        )

    def find_latest_test_dir(self) -> None:
        """Find the latest test directory to continue from."""
        if self.checkpoints_dir.exists():
            test_dirs = [
                d for d in self.checkpoints_dir.iterdir() if d.name.startswith("test_")
            ]
            if test_dirs:
                latest_test_num = max(int(d.name.split("_")[-1]) for d in test_dirs)
                self.config.max_test_i = latest_test_num + 1
            else:
                self.config.max_test_i = 0

    def custom_log_creator(self, custom_path: Union[str, Path], custom_str: str):
        """Create a custom logger for Ray."""

        def logger_creator(config):
            logdir = Path(custom_path) / custom_str
            logdir.mkdir(exist_ok=True, parents=True)
            return UnifiedLogger(config, logdir, loggers=None)

        return logger_creator

    def build_trainer(self, config_index: int, test_number: int) -> PPO:
        """Build a PPO trainer with given configuration."""
        training_config = self.full_config["training"]

        return (
            PPOConfig()
            .environment(**training_config.get("environment", {}))
            .env_runners(**training_config.get("env_runners", {}))
            .checkpointing(**training_config.get("checkpointing", {}))
            .debugging(
                logger_creator=self.custom_log_creator(
                    self.log_dir / f"test_{test_number}", f"config_{config_index}"
                )
            )
            .training(**training_config.get("training", {}))
            .framework(**training_config.get("framework", {}))
            .resources(**training_config.get("resources", {}))
            .evaluation(**training_config.get("evaluation", {}))
            .fault_tolerance(**training_config.get("fault_tolerance", {}))
            .callbacks(callbacks_class=CassieEnvCallback)
        ).build()

    def evaluate(
        self, trainer: PPO, env: Env, epoch: int, config_index: int, test_dir: Path
    ) -> None:
        """Evaluate the current policy and save a video."""
        # Reset environment
        env.reset()
        obs, _ = env.reset()
        done = False
        frames = []
        steps = 0
        render_step = 0  # Track rendered frames

        # Get RL module
        rl_module = trainer.get_module()

        render_fps = self.full_config["run"]["render_fps"]
        sim_fps = self.full_config["training"]["environment"]["env_config"]["sim_fps"]
        # Removed: skip_every = sim_fps // render_fps

        # Ensure fps are positive to avoid division by zero or unexpected behavior
        if render_fps <= 0 or sim_fps <= 0:
            logger.warning("render_fps or sim_fps is non-positive. Skipping rendering.")
            render_fps = 0  # Effectively disable rendering

        # Run episode
        while not done:
            # Get model outputs
            model_outputs = rl_module.forward_inference(
                {
                    "obs": torch.from_numpy(obs.astype(np.float32)).unsqueeze(0),
                }
            )

            # Extract action
            action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
            greedy_action = action_dist_params[: env.action_space.low.shape[0]]

            # Step environment
            obs, _, done, _, _ = env.step(greedy_action)

            # Render and save frame based on time comparison
            # Check if it's time to render the next frame
            should_render = False
            if render_fps > 0:
                # Condition ensures rendering happens at the correct frequency
                if steps * render_fps >= render_step * sim_fps:
                    should_render = True
                    render_step += 1

            if should_render:
                frame = env.render()
                frames.append(frame)

            steps += 1

        # Log episode results
        logger.info("Episode finished after {} steps", steps)
        logger.info("Episode done reason: {}", env.isdone)

        # Save video
        video_path = test_dir / f"config_{config_index}/run_{epoch}.mp4"
        media.write_video(video_path, frames, fps=min(render_fps, sim_fps))
        logger.info("Test saved at {}", video_path)

    def load_checkpoint(self, trainer: PPO, config_index: int) -> bool:
        """Attempt to load the latest checkpoint."""
        if self.config.clean_run:
            return False

        # Find most recent run directories
        runs = sorted(
            Path.glob(self.checkpoints_dir, "test_*"),
            key=os.path.getmtime,
            reverse=True,
        )

        for run in runs:
            if (run / f"config_{config_index}").exists():
                run_config = run / f"config_{config_index}"
            else:
                run_config = run

            logger.info("Checking run {}", run_config)

            # Find checkpoints in this run
            checkpoints = sorted(
                Path.glob(run_config, "checkpoint_*"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )

            # Try to load the latest checkpoint
            for checkpoint in reversed(checkpoints):
                try:
                    # Use absolute path without URI format
                    checkpoint_path = os.path.abspath(checkpoint)
                    trainer.restore(checkpoint_path)
                    logger.info("Checkpoint loaded from {}", checkpoint_path)
                    return True
                except Exception as e:
                    logger.error("Error loading checkpoint: {}", e)

        logger.info("No valid checkpoint found, starting from scratch")
        return False

    def train_and_evaluate(
        self, hyper_configs: List[Optional[Dict[str, Any]]] = None
    ) -> List[float]:
        """Train and evaluate the model with given hyperparameter configurations."""
        if hyper_configs is None:
            hyper_configs = [None]

        test_n = self.config.max_test_i
        metrics = []

        # Prepare results directory
        ray_results_dir = self.log_dir / f"test_{test_n}"
        ray_results_dir.mkdir(exist_ok=True, parents=True)

        for i, hyper_config in enumerate(hyper_configs):
            # Apply hyperparameter configuration if provided
            env_config = self.full_config["training"]["environment"]["env_config"]
            if hyper_config is not None:
                fill_dict_with_list(hyper_config, env_config)

            logger.info("Training config: {}", env_config)

            # Build trainer
            trainer = self.build_trainer(i, test_n)

            # Load checkpoint if available
            self.load_checkpoint(trainer, i)

            # Create test environment
            env = CassieEnv({**env_config, "is_training": False})
            env.render_mode = "rgb_array"

            # Create directory for this test
            test_dir = self.sim_dir / f"test_{test_n}"
            test_dir.mkdir(exist_ok=True)
            (test_dir / f"config_{i}").mkdir(exist_ok=True)

            # Save config
            with (test_dir / f"config_{i}/config.yaml").open("w") as file:
                yaml.dump(env_config, file)

            # Training loop
            start_epoch = trainer.iteration if hasattr(trainer, "iteration") else 0
            max_epochs = self.full_config["run"].get("epochs", 1000)

            for epoch in range(start_epoch, max_epochs):
                try:
                    # Train for one iteration
                    result = trainer.train()
                    logger.info(
                        "Episode {} Reward Mean {}",
                        epoch,
                        result["env_runners"]["episode_return_mean"],
                    )

                    # Save checkpoint if needed
                    current_time = time.time()
                    if (
                        current_time - self.last_checkpoint_time
                        > self.full_config["run"]["chkpt_freq"]
                    ):
                        self.last_checkpoint_time = current_time
                        # Create checkpoint directory
                        checkpoint_dir = (
                            self.checkpoints_dir
                            / f"test_{test_n}/config_{i}/checkpoint_{epoch}"
                        )
                        checkpoint_path = os.path.abspath(checkpoint_dir)
                        trainer.save(checkpoint_path)
                        logger.info("Checkpoint saved at {}", checkpoint_path)

                    # Run evaluation if needed
                    if (
                        current_time - self.last_render_time
                        > self.full_config["run"]["render_every"]
                    ):
                        self.last_render_time = current_time
                        self.evaluate(trainer, env, epoch, i, test_dir)

                except ValueError as e:
                    logger.error("Value error: {}", e)

            # Track metrics
            self.config.max_test_i += 1
            metrics.append(result.get("episode_reward_mean", 0))

        return metrics

    def get_hyperparameter_ranges(self) -> Dict[str, Any]:
        """Get hyperparameter ranges for optimization."""
        hyperparameter_ranges = {}
        env_config = self.full_config["training"]["environment"]["env_config"]

        for key, value in env_config.items():
            if isinstance(value, list) and len(value) == 2:
                # Assume it's a range [min, max]
                range_span = value[1] - value[0]
                hyperparameter_ranges[key] = [
                    [value[0] - range_span / 4, value[0] + range_span / 4],
                    [value[1] - range_span / 4, value[1] + range_span / 4],
                ]
            elif isinstance(value, float):
                # Create a range around the float value
                range_span = abs(value) / 4
                hyperparameter_ranges[key] = [value - range_span, value + range_span]

        return hyperparameter_ranges

    def run_swarm_optimization(self):
        """Run particle swarm optimization for hyperparameter tuning."""
        hyperparameter_ranges = self.get_hyperparameter_ranges()
        logger.info("Hyperparameter ranges: {}", hyperparameter_ranges)

        hyperparameter_bounds = flatten_dict(hyperparameter_ranges)
        min_bounds = [x[0] for x in hyperparameter_bounds.values()]
        max_bounds = [x[1] for x in hyperparameter_bounds.values()]

        # PSO options
        pso_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

        # Initialize optimizer
        optimizer = GlobalBestPSO(
            n_particles=self.full_config["run"]["n_particles"],
            dimensions=len(hyperparameter_bounds),
            bounds=(min_bounds, max_bounds),
            options=pso_options,
        )

        # Run optimization
        best_hyperparameters, best_fitness = optimizer.optimize(
            lambda hyperconfigs: self.train_and_evaluate(hyperconfigs),
            iters=self.full_config["run"]["hyper_par_iter"],
        )

        logger.info("Best hyperparameters: {}", best_hyperparameters)
        logger.info("Best fitness: {}", best_fitness)

        return best_hyperparameters, best_fitness

    def run(self):
        """Run the training process."""
        if self.config.use_swarm:
            self.run_swarm_optimization()
        else:
            self.train_and_evaluate()


def parse_arguments() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Cassie robot")
    parser.add_argument(
        "-cleanrun",
        action="store_true",
        help="Runs without loading the previous simulation",
    )
    parser.add_argument("-simdir", "--simdir", type=str, help="Simulation directory")
    parser.add_argument("-logdir", "--logdir", type=str, help="Log directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--swarm", action="store_true", help="Use swarm optimizer")

    args = parser.parse_args()

    config = TrainingConfig(clean_run=args.cleanrun, use_swarm=args.swarm)

    if args.config:
        config.config_path = args.config

    if args.simdir:
        config.output_dir = args.simdir

    return config


def main():
    """Main entry point."""
    # Parse arguments
    config = parse_arguments()

    # Create and run training manager
    manager = TrainingManager(config)
    manager.run()


if __name__ == "__main__":
    main()
