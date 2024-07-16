import argparse
import logging
from pathlib import Path
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from src.functions import fill_dict_with_list
from src.cassie import CassieEnv
from src.loader import Loader
from loguru import logger
import mediapy as media

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "output"
sim_dir = Path.cwd() / f"{OUTPUT_DIR}/simulations"
models_dir = Path.cwd() / f"{OUTPUT_DIR}/models"
log_dir = Path.cwd() / f"{OUTPUT_DIR}/sb3_logs"


def make_env(env_config, rank, seed=0):
    def _init():
        env = CassieEnv(env_config)
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, video_folder, video_length=500):
        super(VideoRecorderCallback, self).__init__()
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.video_length = video_length

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger.info("Recording video...")
        frames = []
        obs = self.eval_env.reset()[0]
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, _ = self.eval_env.step(action)
            frame = self.eval_env.render()
            frames.append(frame)
            if done:
                break
        video_path = os.path.join(self.video_folder, f"video_{self.num_timesteps}.mp4")
        media.write_video(video_path, frames, fps=self.eval_env.metadata["render_fps"])
        logger.info(f"Video saved at {video_path}")


def train_and_evaluate(config, hyper_configs=None, clean_run=False):
    training_config = config["training"]
    env_config = training_config["env"]["env_config"]

    if hyper_configs is not None:
        fill_dict_with_list(hyper_configs, env_config)

    logger.info(f"Training config: {env_config}")

    # Create vectorized environment
    env = SubprocVecEnv(
        [make_env(env_config, i) for i in range(training_config.get("n_envs", 4))]
    )

    # Create eval environment
    eval_env = Monitor(CassieEnv({**env_config, "is_training": False}))

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["run"]["chkpt_freq"],
        save_path=models_dir,
        name_prefix="cassie_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_model",
        log_path=log_dir,
        eval_freq=config["run"]["sim_freq"],
        deterministic=True,
        render=False,
    )

    video_recorder_callback = VideoRecorderCallback(
        eval_env, video_folder=sim_dir, video_length=500
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        **training_config.get("training", {}),
    )

    if not clean_run:
        # Try to load the latest model
        try:
            latest_model = max(
                Path(models_dir).glob("cassie_model_*.zip"), key=os.path.getctime
            )
            model = PPO.load(latest_model, env=env)
            logger.info(f"Loaded model from {latest_model}")
        except ValueError:
            logger.info("No existing model found. Starting from scratch.")

    # Train the model
    model.learn(
        total_timesteps=config["run"].get("total_timesteps", 1000000),
        callback=[checkpoint_callback, eval_callback, video_recorder_callback],
    )

    # Final save
    model.save(f"{models_dir}/cassie_final_model")

    return model


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

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    if args.logdir is not None:
        log_dir = Path(args.logdir)
    log_dir.mkdir(exist_ok=True, parents=True)

    if args.simdir is not None:
        sim_dir = Path(args.simdir)
    sim_dir.mkdir(exist_ok=True, parents=True)

    models_dir.mkdir(exist_ok=True, parents=True)

    config_path = args.config if args.config else "configs/config_stable.yaml"
    logger.info(f"Config file: {config_path}")

    loader = Loader(logdir=log_dir, simdir=sim_dir)
    config = loader.load_config(config_path)

    clean_run = args.cleanrun

    logger.info(f"Simulation directory: {sim_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Models directory: {models_dir}")

    train_and_evaluate(config, clean_run=clean_run)
