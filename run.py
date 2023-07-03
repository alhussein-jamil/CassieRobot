import argparse
import datetime
import logging as log
import os
import tempfile
from pathlib import Path

import mediapy as media
import ray
import torch
import yaml
from pyswarms.single.global_best import GlobalBestPSO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

import functions as f
from cassie import CassieEnv, MyCallbacks
from loader import Loader

torch.cuda.empty_cache()

log.basicConfig(level=log.DEBUG)


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.date.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
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

    if args.logdir is not None:
        log_dir = args.logdir
    else:
        log_dir = Path.cwd() / "ray_results"
    if args.config is not None:
        config = args.config

    else:
        config = "default_config.yaml"
    print("Config file: {}".format(config))
    ray.init(ignore_reinit_error=True, num_gpus=1)

    clean_run = args.cleanrun

    if clean_run:
        print("Running clean run")
    else:
        runs = Path.glob(log_dir, "cassie_PPO_config*")
        load = False
        for run in list(runs)[::-1]:
            print("Loading run", run)
            checkpoints = Path.glob(run, "checkpoint_*")
            for checkpoint in list(checkpoints)[::-1]:
                print("Loading checkpoint", checkpoint)
                load = True
                break
            if load:
                break
            else:
                print("No checkpoint found here")

    if args.simdir is not None:
        sim_dir = args.simdir
    else:
        sim_dir = "./sims/"
    print("Simulation directory: {}".format(sim_dir))

    print("Log directory: {}".format(log_dir))
    print(os.path.exists(log_dir))

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
    print("Ranges", hyperparameter_ranges)
    hyperparameter_bounds = f.flatten_dict(hyperparameter_ranges)
    # wandb.init(project="Cassie", config=config["training"]["environment"]["env_config"])
    Trainer = PPOConfig
    # Create sim directory if it doesn't exist
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
        # Find the latest directory named test_i in the sim directory
    latest_directory = max(
        [int(d.split("_")[-1]) for d in os.listdir(sim_dir) if d.startswith("test_")],
        default=0,
    )
    max_test_i = latest_directory + 1

    # Define video codec and framerate
    fps = config["run"]["sim_fps"]
    print("FPS: {}".format(fps))
    config["run"]["chkpt_freq"] = int(
        1500000 / config["training"]["training"]["train_batch_size"]
    )
    config["run"]["sim_freq"] = int(
        1500000 / config["training"]["training"]["train_batch_size"]
    )

    checkpoint_frequency = config["run"]["chkpt_freq"]
    simulation_frequency = config["run"]["sim_freq"]

    print("Checkpoint frequency: {}".format(checkpoint_frequency))
    print("Simulation frequency: {}".format(simulation_frequency))


    def train_and_evaluate(config, hyper_configs=[None]):
        global max_test_i
        # Create folder for test
        test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
        os.makedirs(test_dir, exist_ok=True)

        metrics = []
        for i, hyper_config in enumerate(hyper_configs):
            training_config = config["training"]

            if hyper_config is not None:
                f.fill_dict_with_list(
                    hyper_config, training_config["environment"]["env_config"]
                )
            print("Training config ", training_config["environment"]["env_config"])

            checkpoint_path = "Does NOT exist"
            trainer = (
                Trainer()
                .environment(**training_config.get("environment", {}))
                .rollouts(**training_config.get("rollouts", {}))
                .checkpointing(**training_config.get("checkpointing", {}))
                .debugging(
                    logger_creator=custom_log_creator(
                        log_dir, "cassie_PPO_config_{}_".format(i)
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

            # Build trainer
            if not clean_run:
                if load:
                    trainer.restore(checkpoint)

            print("Creating test environment")
            env = CassieEnv({**config["training"]["environment"]["env_config"], **{"is_training": False}})
            env.render_mode = "rgb_array"

            os.mkdir(os.path.join(test_dir, "config_{}".format(i)))
            # save config
            with open(
                os.path.join(test_dir, "config_{}".format(i), "config.yaml"), "w"
            ) as file:
                yaml.dump(config["training"]["environment"]["env_config"], file)

            for epoch in range(
                trainer.iteration if hasattr(trainer, "iteration") else 0,
                config["run"].get("epochs", 1000),
            ):
                # Train for one iteration
                result = trainer.train()
                print(
                    "Episode {} Reward Mean {} Distance {} ".format(
                        epoch,
                        result["episode_reward_mean"],
                        result["custom_metrics"]["custom_metrics_distance_mean"],
                    )
                )

                if result["episode_len_mean"] < 4:
                    break

                # Save model every 10 epochs
                if epoch % checkpoint_frequency == 0:
                    checkpoint_path = trainer.save()
                    print("Checkpoint saved at", checkpoint_path)

                # Run a test every 20 epochs
                if epoch % simulation_frequency == 0:
                    evaluate(trainer, env, epoch, i, test_dir)

            max_test_i += 1
            metrics.append(result["episode_reward_mean"])
        return metrics

    def evaluate(trainer, env, epoch, i, test_dir):
        # Make a steps counter
        steps = 0

        # Run test
        video_path = os.path.join(
            test_dir + "/config_{}".format(i), "run_{}.mp4".format(epoch)
        )
        filterfn = trainer.workers.local_worker().filters["default_policy"]
        env.reset()
        obs = env.reset()[0]
        done = False
        frames = []

        while not done:
            # Increment steps
            steps += 1
            obs = filterfn(obs)
            action = trainer.compute_single_action(obs)
            obs, _, done, _, _ = env.step(action)
            frame = env.render()
            frames.append(frame)

        # Save video
        media.write_video(video_path, frames, fps=fps)
        print("Test saved at", video_path)

    if args.swarm:
        # Define PSO options
        pso_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        min_bounds, max_bounds = [x[0] for x in hyperparameter_bounds.values()], [
            x[1] for x in hyperparameter_bounds.values()
        ]
        optimizer = GlobalBestPSO(
            n_particles=config["run"]["n_particles"],
            dimensions=len(hyperparameter_bounds),
            bounds=(min_bounds, max_bounds),
            options=pso_options,
        )
        best_hyperparameters, best_fitness = optimizer.optimize(
            lambda hyperconfigs: train_and_evaluate(config, hyper_configs=hyperconfigs),
            iters=config["run"]["hyper_par_iter"],
        )
        print("Best hyperparameters", best_hyperparameters)
        print("Best fitness", best_fitness)
    else:
        train_and_evaluate(config)
