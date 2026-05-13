import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import ray
import yaml
from loguru import logger
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from src.env.callbacks import CassieEnvCallback
from src.env.cassie import CassieEnv
from src.training.evaluation import Evaluator
from src.training.loader import Loader
from src.training.utils import fill_dict_with_list


class TrainingManager:
    """Manages the training and evaluation process for Cassie robot."""

    def __init__(self, output_dir: str, config_path: str, clean_run: bool = False):
        self.clean_run = clean_run
        self.last_checkpoint_time = time.time() - np.inf
        self.last_render_time = time.time() - np.inf

        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.sim_dir = self.output_dir / "simulations"
        self.sim_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        self.log_dir = self.output_dir / "ray_results"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.max_test_i = self._find_latest_test_num()

        # Initialize Ray
        ray.init(ignore_reinit_error=True, num_gpus=1)
        register_env("cassie-v0", lambda cfg: CassieEnv(cfg))

        # Load configuration
        self.loader = Loader(logdir=self.log_dir, simdir=self.sim_dir)
        self.full_config = self.loader.load_config(config_path)

        self.full_config["training"]["env_runners"]["num_env_runners"] = (
            multiprocessing.cpu_count()
        )
        # Rollout workers do CPU-only inference well; allocating a fractional
        # GPU per worker forces every one to initialize CUDA, which is pure
        # overhead and contends for VRAM with the learner.
        self.full_config["training"]["env_runners"]["num_gpus_per_env_runner"] = 0

        # Initialize evaluator
        self.evaluator = Evaluator(self.full_config.get("run", {}))

    def _find_latest_test_num(self) -> int:
        """Find the next test directory number."""
        if self.checkpoints_dir.exists():
            test_dirs = [
                d for d in self.checkpoints_dir.iterdir() if d.name.startswith("test_")
            ]
            if test_dirs:
                return max(int(d.name.split("_")[-1]) for d in test_dirs) + 1
        return 0

    def _custom_log_creator(self, custom_path: Union[str, Path], custom_str: str):
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
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(**training_config.get("environment", {}))
            .env_runners(**training_config.get("env_runners", {}))
            .checkpointing(**training_config.get("checkpointing", {}))
            .debugging(
                logger_creator=self._custom_log_creator(
                    self.log_dir / f"test_{test_number}", f"config_{config_index}"
                )
            )
            .training(**training_config.get("training", {}))
            .framework(**training_config.get("framework", {}))
            .resources(**training_config.get("resources", {}))
            .evaluation(**training_config.get("evaluation", {}))
            .fault_tolerance(**training_config.get("fault_tolerance", {}))
            .callbacks(callbacks_class=CassieEnvCallback)
        ).build_algo()

    def load_checkpoint(self, trainer: PPO, config_index: int) -> bool:
        """Attempt to load the latest checkpoint."""
        if self.clean_run:
            return False

        runs = sorted(
            Path.glob(self.checkpoints_dir, "test_*"),
            key=os.path.getmtime,
            reverse=True,
        )

        for run in runs:
            run_config = (
                run / f"config_{config_index}"
                if (run / f"config_{config_index}").exists()
                else run
            )
            logger.info("Checking run {}", run_config)

            checkpoints = sorted(
                Path.glob(run_config, "checkpoint_*"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )

            for checkpoint in reversed(checkpoints):
                try:
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

        test_n = self.max_test_i
        metrics = []

        ray_results_dir = self.log_dir / f"test_{test_n}"
        ray_results_dir.mkdir(exist_ok=True, parents=True)

        for i, hyper_config in enumerate(hyper_configs):
            env_config = self.full_config["training"]["environment"]["env_config"]
            if hyper_config is not None:
                fill_dict_with_list(hyper_config, env_config)

            logger.info("Training config: {}", env_config)

            trainer = self.build_trainer(i, test_n)
            self.load_checkpoint(trainer, i)

            env = CassieEnv({**env_config, "is_training": False})
            env.render_mode = "rgb_array"

            test_dir = self.sim_dir / f"test_{test_n}"
            test_dir.mkdir(exist_ok=True)
            (test_dir / f"config_{i}").mkdir(exist_ok=True)

            with (test_dir / f"config_{i}/config.yaml").open("w") as file:
                yaml.dump(env_config, file)

            start_epoch = trainer.iteration if hasattr(trainer, "iteration") else 0
            max_epochs = self.full_config["run"].get("epochs", 1000)
            result = None

            for epoch in range(start_epoch, max_epochs):
                try:
                    result = trainer.train()
                    logger.info(
                        "Episode {} Reward Mean {}",
                        epoch,
                        result["env_runners"]["episode_return_mean"],
                    )

                    current_time = time.time()
                    if (
                        current_time - self.last_checkpoint_time
                        > self.full_config["run"]["chkpt_freq"]
                    ):
                        self.last_checkpoint_time = current_time
                        checkpoint_dir = (
                            self.checkpoints_dir
                            / f"test_{test_n}/config_{i}/checkpoint_{epoch}"
                        )
                        trainer.save(os.path.abspath(checkpoint_dir))
                        logger.info("Checkpoint saved at {}", checkpoint_dir)

                    if (
                        current_time - self.last_render_time
                        > self.full_config["run"]["render_every"]
                    ):
                        self.last_render_time = current_time
                        sim_fps = env_config.get("sim_fps", 40)
                        self.evaluator.evaluate(
                            trainer, env, epoch, i, test_dir, sim_fps
                        )

                except ValueError as e:
                    logger.error("Value error: {}", e)

            self.max_test_i += 1
            metrics.append(result.get("episode_reward_mean", 0) if result else 0)

        return metrics

    def get_hyperparameter_ranges(self) -> Dict[str, Any]:
        """Get hyperparameter ranges for optimization."""
        hyperparameter_ranges = {}
        env_config = self.full_config["training"]["environment"]["env_config"]

        for key, value in env_config.items():
            if isinstance(value, list) and len(value) == 2:
                range_span = value[1] - value[0]
                hyperparameter_ranges[key] = [
                    [value[0] - range_span / 4, value[0] + range_span / 4],
                    [value[1] - range_span / 4, value[1] + range_span / 4],
                ]
            elif isinstance(value, float):
                range_span = abs(value) / 4
                hyperparameter_ranges[key] = [value - range_span, value + range_span]

        return hyperparameter_ranges
