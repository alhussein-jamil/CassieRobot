import argparse
import logging as log

from pathlib import Path
from typing import Any
import os
import mediapy as media
import numpy as np
import ray
import torch
import yaml
import gymnasium as gym
from pyswarms.single.global_best import GlobalBestPSO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from src.functions import fill_dict_with_list, flatten_dict
from src.cassie import CassieEnv, MyCallbacks
from src.loader import Loader
from loguru import logger

torch.cuda.empty_cache()

log.basicConfig(level=log.DEBUG)

OUTPUT_DIR = "output"
max_test_i = 0
checkpoint = None
load = False
sim_dir = Path.cwd() / f"{OUTPUT_DIR}/simulations"
checkpoints_dir = Path.cwd() / f"{OUTPUT_DIR}/checkpoints"
log_dir = Path.cwd() / f"{OUTPUT_DIR}/ray_results"


def build_trainer(config: dict[str, Any], config_n: int, test_n: int) -> PPO:
    training_config = config["training"]
    trainer = (
        PPOConfig()
        .environment(**training_config.get("environment", {}))
        .env_runners(**training_config.get("env_runners", {}))
        .checkpointing(**training_config.get("checkpointing", {}))
        .debugging(
            logger_creator=custom_log_creator(
                log_dir / f"test_{test_n}", f"config_{config_n}"
            )
            if log_dir is not None
            else None
        )
        .training(**training_config.get("training", {}))
        .framework(**training_config.get("framework", {}))
        .resources(**training_config.get("resources", {}))
        .evaluation(**training_config.get("evaluation", {}))
        .callbacks(callbacks_class=MyCallbacks)
    ).build()
    return trainer


def train_and_evaluate(
    config: dict[str, Any],
    hyper_configs: list[dict[str, Any] | None] = [None],
    # checkpoint: str | Path | None = None,
    clean_run: bool = False,
):
    global max_test_i

    ray_results_dir = Path.cwd() / f"{OUTPUT_DIR}/ray_results/test_{max_test_i}"
    ray_results_dir.mkdir(exist_ok=True)

    metrics = []
    for i, hyper_config in enumerate(hyper_configs):
        training_config = config["training"]

        if hyper_config is not None:
            fill_dict_with_list(
                hyper_config, training_config["environment"]["env_config"]
            )
        logger.info("Training config {}", training_config["environment"]["env_config"])
        test_n = max_test_i
        if clean_run:
            logger.info("Running clean run")
        else:
            # runs = Path.glob(log_dir, "cassie_PPO_config*")
            runs = Path.glob(checkpoints_dir, "test_*")
            load = False
            # sort runs by last modified where the last modified is the first element
            runs = sorted(runs, key=os.path.getmtime, reverse=True)

            for run in list(runs):
                if len(list(run.glob("config_*"))) > 0:
                    run_config = run / f"config_{i}"
                else:
                    run_config = run

                logger.info("Loading run {}", run_config)
                checkpoints = Path.glob(run_config, "checkpoint_*")

                test_n = int(run.stem.split("_")[-1])
                for checkpoint in list(checkpoints)[::-1]:
                    logger.info("Loading checkpoint {}", checkpoint)
                    load = True
                    break
                if load:
                    break
                else:
                    logger.info("No checkpoint found here")

        trainer = build_trainer(config, i, test_n)
        # Build trainer
        if not clean_run:
            if load:
                try:
                    trainer.restore(str(checkpoint).replace("\\", "/"))
                    logger.info("Checkpoint loaded from {}", checkpoint)
                except Exception as e:
                    logger.error(
                        "Error loading checkpoint: {}, starting from scratch", e
                    )

        logger.info("Creating test environment")
        env = CassieEnv(
            {
                **config["training"]["environment"]["env_config"],
                **{"is_training": False},
            }
        )
        env.render_mode = "rgb_array"

        # Create folder for test
        test_dir = Path(sim_dir) / f"test_{test_n}"
        test_dir.mkdir(exist_ok=True)

        (test_dir / f"config_{i}").mkdir(exist_ok=True)

        # save config
        with (test_dir / f"config_{i}/config.yaml").open("w") as file:
            yaml.dump(config["training"]["environment"]["env_config"], file)

        for epoch in range(
            trainer.iteration if hasattr(trainer, "iteration") else 0,
            config["run"].get("epochs", 1000),
        ):
            try:
                # Train for one iteration
                result = trainer.train()
            except ValueError as e:
                logger.error("Value error: {}", e)

            logger.info(
                "Episode {} Reward Mean {}  ",
                epoch,
                result["env_runners"]["episode_reward_mean"],
                # result["episode_reward_mean"],
                # result["custom_metrics"]["custom_metrics_distance_mean"],
            )

            # if result["episode_len_mean"] < 4:
            #     break

            # Save model every 10 epochs
            if epoch % checkpoint_frequency == 0:
                checkpoint_dir_tmp = (
                    checkpoints_dir / f"test_{test_n}/config_{i}/checkpoint_{epoch}"
                )
                trainer.save(checkpoint_dir_tmp)
                logger.info(
                    "Checkpoint saved at {}",
                    checkpoint_dir_tmp,
                )

            # Run a test every 20 epochs
            if epoch % simulation_frequency == 0:
                evaluate(trainer, env, epoch, i, test_dir)

        max_test_i += 1
        metrics.append(result["episode_reward_mean"])
    return metrics


def evaluate(trainer: PPO, env: gym.Env, epoch: int, i: int, test_dir: str | Path):
    # Make a steps counter
    steps = 0

    # Run test
    video_path = Path(test_dir) / f"config_{i}/run_{epoch}.mp4"
    filterfn = trainer.workers.local_worker().filters["default_policy"]
    env.reset()
    obs = env.reset()[0]
    done = False
    frames = []
    fps = env.metadata["render_fps"] // 2
    action = np.zeros(env.action_space.shape[0])
    while not done:
        # Increment steps
        steps += 1
        obs = filterfn(obs)
        try:
            action = trainer.compute_single_action(obs)
        except ValueError as e:
            logger.error("Value error: {}", e)

        obs, _, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
    logger.info("Episode finished after {} steps", steps)
    logger.info("Episode done reason: {}", env.isdone)
    # Save video
    media.write_video(video_path, frames, fps=fps)
    logger.info("Test saved at {}", video_path)


def custom_log_creator(custom_path: str | Path, custom_str: str):
    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = Path(custom_path)  / custom_str
        logdir.mkdir(exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-cleanrun",
        action="store_true",
        help="Runs without loading the previous simulation",
    )
    argparser.add_argument("-simdir", "--simdir", type=str, help="Simulation directory")
    argparser.add_argument("-logdir", "--logdir", type=str, help="Log directory")
    argparser.add_argument("--config", type=str, help="Path to config file")
    argparser.add_argument("--swarm", action="store_true", help="Use swarm optimizer")
    args = argparser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    if args.logdir is not None:
        log_dir = args.logdir
    else:
        log_dir = Path.cwd() / f"{OUTPUT_DIR}/ray_results"
    if args.config is not None:
        config = args.config

    else:
        config = "configs/default_config.yaml"
    logger.info("Config file: {}", config)
    ray.init(ignore_reinit_error=True, num_gpus=1)

    clean_run = args.cleanrun


    if args.simdir is not None:
        sim_dir = args.simdir
    else:
        sim_dir = Path.cwd() / f"{OUTPUT_DIR}/simulations"

    logger.info("Simulation directory: {}", sim_dir)
    logger.info("Log directory: {}", log_dir)

    register_env("cassie-v0", lambda config: CassieEnv(config))

    loader = Loader(logdir=log_dir, simdir=sim_dir)

    config = loader.load_config(config)

    hyperparameter_ranges = {}

    for key, value in config["training"]["environment"]["env_config"].items():
        if isinstance(value, list):
            hyperparameter_ranges[key] = [
                [
                    value[0] - (value[1] - value[0]) / 4,
                    value[0] + (value[1] - value[0]) / 4,
                ],
                [
                    value[1] - (value[1] - value[0]) / 4,
                    value[1] + (value[1] - value[0]) / 4,
                ],
            ]
        elif isinstance(value, float):
            hyperparameter_ranges[key] = (
                [value - value / 4, value + value / 4]
                if value >= 0
                else [value + value / 4, value - value / 4]
            )
    logger.info("Hyperparameters Ranges", hyperparameter_ranges)
    hyperparameter_bounds = flatten_dict(hyperparameter_ranges)

    # Uncomment to use wandb
    # wandb.init(project="Cassie", config=config["training"]["environment"]["env_config"])

    # Create sim directory if it doesn't exist
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
        # Find the latest directory named test_i in the sim directory
    latest_directory = max(
        [int(d.split("_")[-1]) for d in os.listdir(checkpoints_dir) if d.startswith("test_")],
        default=0,
    )
    max_test_i = latest_directory + 1

    # Define video codec and framerate
    fps = config["run"]["sim_fps"]
    logger.info("FPS: {}", fps)

    config["run"]["chkpt_freq"] = int(
        3000000 / config["training"]["training"]["train_batch_size"]
    )
    config["run"]["sim_freq"] = int(
        1500000 / config["training"]["training"]["train_batch_size"]
    )

    checkpoint_frequency = config["run"]["chkpt_freq"]
    simulation_frequency = config["run"]["sim_freq"]

    logger.info("Checkpoint frequency: {}", checkpoint_frequency)
    logger.info("Simulation frequency: {}", simulation_frequency)

    if args.swarm:
        # Define PSO options
        pso_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        min_bounds, max_bounds = (
            [x[0] for x in hyperparameter_bounds.values()],
            [x[1] for x in hyperparameter_bounds.values()],
        )
        optimizer = GlobalBestPSO(
            n_particles=config["run"]["n_particles"],
            dimensions=len(hyperparameter_bounds),
            bounds=(min_bounds, max_bounds),
            options=pso_options,
        )
        best_hyperparameters, best_fitness = optimizer.optimize(
            lambda hyperconfigs: train_and_evaluate(
                config, hyper_configs=hyperconfigs
            ),
            iters=config["run"]["hyper_par_iter"],
        )
        print("Best hyperparameters", best_hyperparameters)
        print("Best fitness", best_fitness)
    else:
        train_and_evaluate(config, clean_run=clean_run)
