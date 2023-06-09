import argparse
import datetime
import logging as log
import os
import tempfile
from pathlib import Path

import mediapy as media
import numpy as np
import ray
import torch
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
import functions as f
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

# import wandb
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

    args = argparser.parse_args()

    if args.logdir is not None:
        log_dir = args.logdir
    else:
        log_dir = Path.home() / "ray_results"
    if args.config is not None:
        config = args.config

    else:
        config = "default_config.yaml"
    log.info("Config file: {}".format(config))
    ray.init(ignore_reinit_error=True, num_gpus=1)

    clean_run = args.cleanrun

    if clean_run:
        log.info("Running clean run")
    else:
        runs = Path.glob(log_dir, "PPO_cassie-v0*")
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
    log.info("Simulation directory: {}".format(sim_dir))

    log.info("Log directory: {}".format(log_dir))
    log.info(os.path.exists(log_dir))

    register_env("cassie-v0", lambda config: CassieEnv(config))

    loader = Loader(logdir=log_dir, simdir=sim_dir)

    config = loader.load_config(config)
    hyperparameter_ranges = {}
    
    for key, value in config["training"]["environment"]["env_config"].items():
        if isinstance(value, list):

            hyperparameter_ranges[key] = [tune.uniform(
            
                value[0] - (value[1] - value[0]) / 2,
                value[0] + (value[1] - value[0]) / 2)
            ,
            tune.uniform(
                value[1] - (value[1] - value[0]) / 2,
                value[1] + (value[1] - value[0]) / 2
            )
            ]
        elif isinstance(value, float):
            hyperparameter_ranges[key] = tune.uniform(value - 0.1 * value, value + 0.1 * value)
    print(hyperparameter_ranges)
    print("flattened: ", f.flatten_dict(hyperparameter_ranges))
    flattened = f.flatten_dict(hyperparameter_ranges)
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

    # Create folder for test
    test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
    os.makedirs(test_dir, exist_ok=True)

    # Define video codec and framerate
    fps = config["run"]["sim_fps"]
    print("fps", fps)

    checkpoint_frequency = config["run"]["chkpt_freq"]
    simulation_frequency = config["run"]["sim_freq"]
    def train_and_evaluate(hyper_config, config, max_test_i, i=0):

        import functions as f

        training_config = config["training"]
        print("hyper_config", list(hyper_config.values()))
        print("training_config", training_config["environment"]["env_config"])
        f.fill_dict_with_list(list(hyper_config.values()), training_config["environment"]["env_config"])
        trainer = (
            Trainer()
            .environment(**training_config.get("environment", {}))
            .rollouts(**training_config.get("rollouts", {}))
            .checkpointing(**training_config.get("checkpointing", {}))
            .debugging(
                logger_creator=custom_log_creator(
                    args.logdir, "cassie_PPO_config_{}_".format(i)
                )
                if args.logdir is not None
                else None
            )
            .training(**training_config.get("training", {}))
            .framework(**training_config.get("framework", {}))
            .resources(**training_config.get("resources", {}))
            .evaluation(**training_config.get("evaluation", {}))
            .callbacks(callbacks_class=MyCallbacks)
        )

        trainer = trainer.build()

        # Build trainer
        if not clean_run:
            if load:
                trainer.restore(checkpoint)

        print("Creating test environment")
        env = CassieEnv(config["training"]["environment"])
        env.render_mode = "rgb_array"

        os.mkdir(os.path.join(test_dir, "config_{}".format(i)))
        # save config
        with open(
            os.path.join(test_dir, "config_{}".format(i), "config.yaml"), "w"
        ) as f:
            yaml.dump(config, f)

        for epoch in range(
            trainer.iteration if hasattr(trainer, "iteration") else 0,
            config["run"].get("epochs", 1000),
        ):
            # Train for one iteration
            result = trainer.train()
            # wandb.log(
            #     {
            #         "iteration": i,
            #         **training_config["environment"]["env_config"],
            #         "loss": result["episode_reward_mean"],
            #     }
            # )
            print(
                "Episode {} Reward Mean {} Q_lef_frc {} Q_left_spd {}".format(
                    epoch,
                    result["episode_reward_mean"],
                    result["custom_metrics"]["custom_quantities_q_left_frc_mean"],
                    result["custom_metrics"]["custom_quantities_q_left_spd_mean"],
                )
            )

            # Save model every 10 epochs
            if epoch % checkpoint_frequency == 0:
                checkpoint_path = trainer.save()
                print("Checkpoint saved at", checkpoint_path)

            # Run a test every 20 epochs
            if epoch % simulation_frequency == 0:
                evaluate(trainer, env, epoch, i)

                max_test_i += 1

        return  result["episode_reward_mean"]

    def evaluate(trainer, env, epoch, i):
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

    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=2,
        hyperparam_mutations=flattened,
        
    )
    N = 6
    analysis = tune.run(
        lambda hyper_config : train_and_evaluate(hyper_config, config ,max_test_i),
        config=flattened,
        scheduler=pbt_scheduler,
        num_samples=config["run"]["hyper_par_iter"],
        resources_per_trial = tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * N)

    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_config = best_trial.config


