from ray.tune.logger import UnifiedLogger
# from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from loader import Loader
import mediapy as media
import os
from cassie import CassieEnv, MyCallbacks
from ray.tune.registry import register_env
import argparse
from caps import *
import logging as log
import ray
from pathlib import Path
from ray.rllib.models import ModelCatalog


log.basicConfig(level=log.DEBUG)

if __name__ == "__main__":
    # To call the function I wan to use the following command: python run.py
    # -clean --simdir="" --logdir=""
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
        help="Simulation directory")
    argparser.add_argument(
        "-logdir",
        "--logdir",
        type=str,
        help="Log directory")
    argparser.add_argument(
        "-caps", action="store_true", help="Uses CAPS regularization"
    )
    argparser.add_argument("--config", type=str, help="Path to config file")
    argparser.add_argument("--simfreq", type=int, help="Simulation frequency")
    argparser.add_argument(
        "--checkfreq",
        type=int,
        help="Checkpoint frequency")
    argparser.add_argument(
        "-configIsDict", action="store_true", help="Config is a dictionary"
    )
    argparser.add_argument(
        "-oldImplementation",
        action="store_true",
        help="Uses old implementation of Trainers",
    )
    
    args = argparser.parse_args()

    # flush gpu memory

    old_implementation = args.oldImplementation

    import torch

    torch.cuda.empty_cache()

    ModelCatalog.register_custom_model(
        "caps_loss",
        CAPSTorchPolicy
    )
    is_dict = args.configIsDict
    ray.init(ignore_reinit_error=True,num_gpus=1)
    if args.simfreq is not None:
        simulation_frequency = args.simfreq
        log.info("Simulation frequency: {}".format(simulation_frequency))
    else:
        sim_freq = 10

    if args.checkfreq is not None:
        checkpoint_frequency = args.checkfreq
        log.info("Checkpoint frequency: {}".format(simulation_frequency))
    else:
        check_freq = 5

    clean_run = args.cleanrun
    load = False
    if clean_run:
        log.info("Running clean run")
    else:
        runs = Path.glob( Path.home() / "ray_results" ,"PPO_cassie-v0*")

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
        log.info("Simulation directory: {}".format(sim_dir))
    else:
        sim_dir = "./sims/"

    if args.logdir is not None:
        log_dir = args.logdir
    else:
        log_dir = "C:/Users/Ajvendetta/ray_results"

    log.info("Log directory: {}".format(log_dir))
    log.info(os.path.exists(log_dir))
    register_env("cassie-v0", lambda config: CassieEnv(config))

    loader = Loader(logdir=log_dir, simdir=sim_dir)
    config = "config.yaml"

    if args.config is not None:
        config = args.config

    config = loader.load_config(config)
    Trainer = PPOConfig

    
    # if not args.caps:
    #     log.info("Running without CAPS regularization")
    # else:
    #     log.info("Running with CAPS regularization")
    #     config["model"]["custom_loss"] = CAPSTorchPolicy.loss

    if is_dict:
        trainer = Trainer().from_dict({**config, "callbacks": MyCallbacks})

    else:
        splitted = loader.split_config(config)

        trainer = (
            Trainer()
            .environment(**splitted.get("environment", {}))
            .rollouts(**splitted.get("rollouts", {}))
            .checkpointing(**splitted.get("checkpointing", {}))
            .debugging(**splitted.get("debugging", {}))
            .training(**splitted.get("training", {}))
            .framework(**splitted.get("framework", {}))
            .resources(**splitted.get("resources", {}))
            .evaluation(**splitted.get("evaluation", {}))
            .callbacks(callbacks=MyCallbacks)
        )
        
    trainer = trainer.build()


    if not args.caps:
        log.info("Running without CAPS regularization")
    else:
        log.info("Running with CAPS regularization")
        trainer.get_policy().loss= CAPSTorchPolicy.loss
    #build trainer

    if not clean_run: #and weights is not None:
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
        default=0,
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
        #get the current filter params
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
            # make a steps counter
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
