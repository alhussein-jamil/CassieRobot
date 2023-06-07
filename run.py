from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms.ppo import PPOConfig
from loader import Loader
import mediapy as media
import os
from cassie import CassieEnv, MyCallbacks
from ray.tune.registry import register_env
import argparse
from caps import CapsModel
import logging as log
import ray
from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.custom_loss_model import (
    CustomLossModel,
    TorchCustomLossModel,
)
import datetime
import tempfile

import torch

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

ModelCatalog.register_custom_model("caps_model", CapsModel)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-cleanrun",
        action="store_true",
        help="Runs without loading the previous simulation",
    )
    argparser.add_argument(
        "-simdir",
        "--simdir",
        type=str,
        help="Simulation directory"
    )
    argparser.add_argument(
        "-logdir",
        "--logdir",
        type=str,
        help="Log directory"
    )
    argparser.add_argument(
        "-caps",
        action="store_true",
        help="Uses CAPS regularization"
    )
    argparser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )

    args = argparser.parse_args()

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
        runs = Path.glob(Path.home() / "ray_results", "PPO_cassie-v0*")

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

    if args.logdir is not None:
        log_dir = args.logdir
    else:
        log_dir = "default_logdir"
    log.info("Log directory: {}".format(log_dir))
    log.info(os.path.exists(log_dir))

    register_env("cassie-v0", lambda config: CassieEnv(config))

    loader = Loader(logdir=log_dir, simdir=sim_dir)

    config = loader.load_config(config)
    
    Trainer = PPOConfig

    training_config = config["training"]

    print("env_config", training_config.get("environment", {})["env_config"])

    trainer = (
        Trainer()
        .environment(**training_config.get("environment", {}))
        .rollouts(**training_config.get("rollouts", {}))
        .checkpointing(**training_config.get("checkpointing", {}))
        .debugging(logger_creator= custom_log_creator(args.logdir , "cassie_PPO") if args.logdir is not None else None)
        .training(**training_config.get("training", {}))
        .framework(**training_config.get("framework", {}))
        .resources(**training_config.get("resources", {}))
        .evaluation(**training_config.get("evaluation", {}))
        .callbacks(callbacks_class=MyCallbacks)
    )

    trainer = trainer.build()

    # Build trainer
    if not clean_run:
        if(load):
            trainer.restore(checkpoint)

    # Define video codec and framerate
    fps = config["run"]["sim_fps"]

    # Training loop
    max_test_i = 0
    checkpoint_frequency = config["run"]["chkpt_freq"]
    simulation_frequency = config["run"]["sim_freq"]
    env = CassieEnv({})
    env.render_mode = "rgb_array"

    # Create sim directory if it doesn't exist
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    # Find the latest directory named test_i in the sim directory
    latest_directory = max(
        [int(d.split("_")[-1]) for d in os.listdir(sim_dir) if d.startswith("test_")],
        default=0
    )
    max_test_i = latest_directory + 1

    # Create folder for test
    test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
    os.makedirs(test_dir, exist_ok=True)

    # Set initial iteration count
    i = trainer.iteration if hasattr(trainer, "iteration") else 0

    while True:
        # Train for one iteration
        result = trainer.train()
        i += 1
        print(
            "Episode {} Reward Mean {} Q_lef_frc {} Q_left_spd {}".format(
                i,
                result["episode_reward_mean"],
                result["custom_metrics"]["custom_quantities_q_left_frc_mean"],
                result["custom_metrics"]["custom_quantities_q_left_spd_mean"]
            )
        )

        # Save model every 10 epochs
        if i % checkpoint_frequency == 0:
            checkpoint_path = trainer.save()
            print("Checkpoint saved at", checkpoint_path)

        # Run a test every 20 epochs
        if i % simulation_frequency == 0:
            # Make a steps counter
            steps = 0

            # Run test
            video_path = os.path.join(test_dir, "sim_{}.mp4".format(i))
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

            # Increment test index
            max_test_i += 1
