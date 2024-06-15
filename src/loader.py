import contextlib
import logging as log
import yaml
from pathlib import Path
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.algorithms import AlgorithmConfig


class Loader:
    def __init__(self, logdir="ray_results/", simdir="./sims/"):
        self.logdir = Path(logdir)
        self.simdir = Path(simdir)

        # register the policy
        POLICIES["CAPSTorchPolicy"] = "caps"

    def find_checkpoint(self, trainer_name="PPO"):
        print("Trainer name ", trainer_name)
        checkpoint_path = None
        # load the trainer from the latest checkpoint if exists
        # get the full directory of latest modified directory in the log_dir
        if self.logdir.exists():
            log.info(f"Log directory exists with path {self.logdir}")
            latest_log_directory = max(
                (
                    d
                    for d in self.logdir.iterdir()
                    if d.is_dir() and d.name.startswith(trainer_name + "_")
                ),
                default=None,
                key=lambda d: d.stat().st_mtime,
            )
            if latest_log_directory is None:
                log.info("No checkpoint found in the log directory")
            else:
                log.info(
                    f"Found log directory with path {latest_log_directory} and latest log directory is {latest_log_directory.name}"
                )
                # check that the folder is not empty
                latest_directory = max(
                    (
                        d
                        for d in latest_log_directory.iterdir()
                        if d.name.startswith("checkpoint")
                    ),
                    default=None,
                    key=lambda d: d.stat().st_mtime,
                )
                if latest_directory is None:
                    log.info("No checkpoint found in the log directory")
                else:
                    checkpoint_path = latest_directory
                    log.info(
                        f"Found checkpoint in the log directory with path {checkpoint_path}"
                    )
                    return str(checkpoint_path)
        return None

    # loads config from the yaml file and returns is as a dictionary
    def load_config(self, path: Path | str) -> dict | None:
        with Path(path).open() as stream:
            with contextlib.suppress(yaml.YAMLError):
                con = yaml.safe_load(stream)
                return con

    def split_config(self, config):
        # split the config into multiple configs
        trainingConfig = config.get("training", {})
        frameworkConfig = config.get("framework", {})
        resourcesConfig = config.get("resources", {})
        evaluationConfig = config.get("evaluation", {})
        environmentConfig = config.get("environment", {})
        rolloutsConfig = config.get("rollouts", {})
        checkpointingConfig = config.get("checkpointing", {})
        debuggingConfig = config.get("debugging", {})
        return {
            "training": trainingConfig,
            "framework": frameworkConfig,
            "resources": resourcesConfig,
            "evaluation": evaluationConfig,
            "environment": environmentConfig,
            "rollouts": rolloutsConfig,
            "checkpointing": checkpointingConfig,
            "debugging": debuggingConfig,
        }

    def recover_weights(
        self,
        TrainerConfig: type[AlgorithmConfig],
        checkpoint_path: str,
        config: dict,
        is_dict=False,
        old_implementation=False,
    ):
        # check that checkpoint path is valid and that it does not end with
        # checkpoint_0
        checkpoint_dir = Path(checkpoint_path)
        if checkpoint_dir.exists() and checkpoint_dir.parent.name.split("_")[-1] != "0":
            # load the a temporary trainer from the checkpoint
            log.info("creating dummy trainer")
            if is_dict:
                if not old_implementation:
                    temp = TrainerConfig().from_dict(config).build()
                else:
                    temp = TrainerConfig(config=config, env="cassie-v0")  # type: ignore

                log.info("dict config")
            else:
                splitted = self.split_config(config)
                if not old_implementation:
                    temp = (
                        TrainerConfig()
                        .environment(**splitted.get("environment", {}))
                        .rollouts(**splitted.get("rollouts", {}))
                        .checkpointing(**splitted.get("checkpointing", {}))
                        .debugging(**splitted.get("debugging", {}))
                        .training(**splitted.get("training", {}))
                        .framework(**splitted.get("framework", {}))
                        .resources(**splitted.get("resources", {}))
                        .evaluation(**splitted.get("evaluation", {}))
                        .build()
                    )
                else:
                    # combine all splitted into one dictionary
                    combined = {
                        **splitted.get("environment", {}),
                        **splitted.get("rollouts", {}),
                        **splitted.get("checkpointing", {}),
                        **splitted.get("debugging", {}),
                        **splitted.get("training", {}),
                        **splitted.get("framework", {}),
                        **splitted.get("resources", {}),
                        **splitted.get("evaluation", {}),
                    }
                    temp = TrainerConfig(config=combined, env="cassie-v0")  # type: ignore
                log.info("generalised config")

            temp.restore(str(checkpoint_path))  # type: ignore

            # Get policy weights
            policy_weights = temp.get_policy().get_weights()  # type: ignore
            # Destroy temp
            temp.stop()  # type: ignore
            del temp  # free memory
            log.info("Weights loaded from checkpoint successfully")
            return policy_weights
        else:
            temp = None
            log.error("can't find checkpoint")
            return None
