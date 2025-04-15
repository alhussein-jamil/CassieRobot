import numpy as np
from scipy import stats
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
def mod(a, b):
    """
    Computes `a mod b` centered around zero.

    The result `r` satisfies:
    - `r = a - k*b` for some integer `k`
    - `-abs(b)/2 <= r < abs(b)/2` if b > 0
    - `-abs(b)/2 < r <= abs(b)/2` if b < 0

    Args:
        a: The dividend. Can be a scalar or NumPy array.
        b: The divisor. Must be non-zero. Can be a scalar or NumPy array (if broadcasting is intended).

    Returns:
        The result of the centered modulo operation. Matches the type of `a / b`.
    """
    # Using the formula: (a + b/2) % b - b/2
    # Ensure floating point division
    b_half = b / 2.0
    # Calculate result using standard modulo operator (%) which works correctly in Numba/NumPy
    # Note: The behavior of % with negative numbers matches Python's definition (sign of divisor).
    result = (a + b_half) % b - b_half
    return result


@nb.jit(nopython=True, cache=True)
def action_dist(
    a: "npt.NDArray[np.float64]",
    b: "npt.NDArray[np.float64]",
    actions_high: "npt.NDArray[np.float64]",
    actions_low: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    diff = a - b

    diff /= actions_high - actions_low
    diff = np.sum(np.square(diff), axis=1)

    return np.sqrt(diff)


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
