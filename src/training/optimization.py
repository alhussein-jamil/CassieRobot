from loguru import logger
from pyswarms.single.global_best import GlobalBestPSO

from src.training.manager import TrainingManager
from src.training.utils import flatten_dict


def run_swarm_optimization(manager: TrainingManager) -> tuple:
    """Run particle swarm optimization for hyperparameter tuning."""
    hyperparameter_ranges = manager.get_hyperparameter_ranges()
    logger.info("Hyperparameter ranges: {}", hyperparameter_ranges)

    hyperparameter_bounds = flatten_dict(hyperparameter_ranges)
    min_bounds = [x[0] for x in hyperparameter_bounds.values()]
    max_bounds = [x[1] for x in hyperparameter_bounds.values()]

    pso_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    optimizer = GlobalBestPSO(
        n_particles=manager.full_config["run"]["n_particles"],
        dimensions=len(hyperparameter_bounds),
        bounds=(min_bounds, max_bounds),
        options=pso_options,
    )

    best_hyperparameters, best_fitness = optimizer.optimize(
        lambda hyperconfigs: manager.train_and_evaluate(hyperconfigs),
        iters=manager.full_config["run"]["hyper_par_iter"],
    )

    logger.info("Best hyperparameters: {}", best_hyperparameters)
    logger.info("Best fitness: {}", best_fitness)

    return best_hyperparameters, best_fitness
