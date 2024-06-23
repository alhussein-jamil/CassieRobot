import numpy as np
from scipy import stats
from .constants import obs_ranges, act_ranges
from typing import TYPE_CHECKING
import numba as nb

if TYPE_CHECKING:
    import numpy.typing as npt


def p_between_von_mises(a, b, kappa, x):
    # Calculate the CDF values for A and B at x
    cdf_a = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * a)
    cdf_b = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * b)

    # Calculate the probability of A < x < B
    p_between = np.abs(cdf_b - cdf_a)

    return p_between


@nb.jit(nopython=True, cache=True)
def action_dist(
    a: "npt.NDArray[np.float64]", b: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    diff = a - b

    diff /= act_ranges[:, 1] - act_ranges[:, 0]
    diff = np.sum(np.square(diff), axis=1)

    return np.sqrt(diff)


@nb.jit(nopython=True, cache=True)
def normalize(name: str, value: float) -> float:
    # normalize the value to be between 0 and 1
    return (value - obs_ranges[name][0]) / (obs_ranges[name][1] - obs_ranges[name][0])


@nb.jit(nopython=True, cache=True)
def von_mises_approx(a, b, kappa, x):
    KappaEQ = 6.072980 * np.log(0.055739 * kappa + 2.365671) + -3.936459
    return 1 / 2 + 1 / 2 * (np.tanh(KappaEQ * np.sin(2 * np.pi * x)))


def apply_f_to_nested_dict(f, nested_dict):
    """
    Applies f to all values in a nested dict
    """
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            apply_f_to_nested_dict(f, v)
        elif isinstance(v, list):
            for i in range(len(v)):
                v[i] = f(v[i])
        elif isinstance(v, float):
            nested_dict[k] = f(v)


def flatten_dict(nested_dict, parent=""):
    """
    Flattens a nested dict
    """
    flat_dict = {}
    if isinstance(nested_dict, list):
        return nested_dict
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, parent + k + "_"))
        elif isinstance(v, list) and isinstance(v[0], list):
            for i in range(len(v)):
                flat_dict[parent + k + "_" + str(i)] = v[i]
        else:
            flat_dict[parent + k] = v
    return flat_dict


@nb.jit(nopython=True, cache=True)
def fill_dict_with_list(list_values, dictionary, index=0):
    """
    Fills a nested dict with a list
    """
    for k, v in dictionary.items():
        if isinstance(v, dict):
            fill_dict_with_list(list_values, v, index)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], float):
                    v[i] = list_values[index]
                    index += 1
        elif isinstance(v, float):
            dictionary[k] = list_values[index]
            index += 1
    return dictionary
