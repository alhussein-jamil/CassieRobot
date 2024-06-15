import os
import logging as log
import yaml
from ray.rllib.algorithms.registry import POLICIES


class Loader:
    def __init__(self, logdir="ray_results/", simdir="./sims/"):
        self.logdir = logdir
        self.simdir = simdir

        # register the policy
        POLICIES["CAPSTorchPolicy"] = "caps"

    def find_checkpoint(self, trainer_name="PPO"):
        print("Trainer name ", trainer_name)
        checkpoint_path = None
        # load the trainer from the latest checkpoint if exists
        # get the full directory of latest modified directory in the log_dir
        if os.path.exists(self.logdir):
            log.info("Log directory exists with path " + self.logdir)
            latest_log_directory = max(
                [
                    d
                    for d in os.listdir(self.logdir)
                    if d.startswith(trainer_name + "_")
                ],
                default=0,
            )
            log.info(
                "Found log directory with path "
                + os.path.join(self.logdir, str(latest_log_directory))
                + " and latest log directory is (if 0 means no logs)"
                + str(latest_log_directory)
            )
            # check that the folder is not empty
            if latest_log_directory == 0:
                log.info("No checkpoint found in the log directory")
            else:
                # get the latest directory in the latest log directory
                latest_directory = max(
                    [
                        d.split("_")[-1]
                        for d in os.listdir(
                            os.path.join(self.logdir, latest_log_directory)
                        )
                        if d.startswith("checkpoint")
                    ],
                    default=0,
                )
                # load the trainer from the latest checkpoint
                checkpoint_path = os.path.join(
                    self.logdir,
                    latest_log_directory,
                    "checkpoint_{}/".format(
                        latest_directory,
                    ),
                )
                log.info(
                    "Found checkpoint in the log directory with path " + checkpoint_path
                )
                return checkpoint_path
        return None

    # loads config from the yaml file and returns is as a dictionary
    def load_config(self, path):
        with open(path, "r") as stream:
            try:
                con = yaml.safe_load(stream)
                print(con)
                return con
            except yaml.YAMLError as exc:
                print(exc)

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
        self, Trainer, checkpoint_path, config, is_dict=False, old_implementation=False
    ):
        # check that checkpoint path is valid and that it does not end with
        # checkpoint_0
        if (
            checkpoint_path is not None
            and checkpoint_path.split("/")[-2].split("_")[-1] != "0"
        ):
            # load the a temporary trainer from the checkpoint
            log.info("creating dummy trainer")
            if is_dict:
                if not old_implementation:
                    temp = Trainer().from_dict(config).build()
                else:
                    temp = Trainer(config=config, env="cassie-v0")

                log.info("dict config")
            else:
                splitted = self.split_config(config)
                if not old_implementation:
                    temp = (
                        Trainer()
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
                    temp = Trainer(config=combined, env="cassie-v0")
                log.info("generalised config")

            temp.restore(checkpoint_path)

            # Get policy weights
            policy_weights = temp.get_policy().get_weights()
            # Destroy temp
            temp.stop()
            del temp  # free memory
            log.info("Weights loaded from checkpoint successfully")
            return policy_weights
        else:
            temp = None
            log.error("can't find checkpoint")
            return None
