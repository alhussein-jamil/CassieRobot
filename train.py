import argparse
import logging

import torch
from loguru import logger

from src.training.manager import TrainingManager
from src.training.optimization import run_swarm_optimization

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger.remove()
logger.add(
    lambda msg: print(msg), colorize=True, format="<level>{level}</level> | {message}"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Cassie robot")
    parser.add_argument(
        "-cleanrun",
        action="store_true",
        help="Run without loading previous simulation",
    )
    parser.add_argument("-simdir", "--simdir", type=str, help="Simulation directory")
    parser.add_argument("-logdir", "--logdir", type=str, help="Log directory")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--swarm", action="store_true", help="Use swarm optimizer")
    return parser.parse_args()


def main():
    args = parse_arguments()

    output_dir = args.simdir if args.simdir else "output"

    manager = TrainingManager(
        output_dir=output_dir,
        config_path=args.config,
        clean_run=args.cleanrun,
    )

    if args.swarm:
        run_swarm_optimization(manager)
    else:
        manager.train_and_evaluate()


if __name__ == "__main__":
    main()
