import numpy as np
import torch
from scipy import stats

import constants as c


def p_between_von_mises(a, b, kappa, x):
    # Calculate the CDF values for A and B at x
    cdf_a = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * a)
    cdf_b = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * b)

    # Calculate the probability of A < x < B
    p_between = np.abs(cdf_b - cdf_a)

    return p_between


def action_dist(a, b):
    diff = a - b

    diff = torch.div(diff, c.act_ranges[:, 1] - c.act_ranges[:, 0])
    diff = torch.sum(torch.square(diff), axis=1)

    return torch.sqrt(diff)


def normalize(name, value):
    # normalize the value to be between 0 and 1
    return (value - c.obs_ranges[name][0]) / (
        c.obs_ranges[name][1] - c.obs_ranges[name][0]
    )


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

def flatten_dict(nested_dict, parent = ""):
    """
    Flattens a nested dict
    """
    flat_dict = {}
    if isinstance(nested_dict, list):
        return nested_dict
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, parent + k + "_"))
        elif isinstance(v, list):
            for i in range(len(v)):
                flat_dict[parent + k + "_" + str(i)] = v[i]
        else:
            flat_dict[parent + k] = v
    return flat_dict

def fill_dict_with_list(l, d, index = 0 ):
    """
    Fills a nested dict with a list
    """
    print(l)
    print(d)
    for k, v in d.items():
        if isinstance(v, dict):
            fill_dict_with_list(l, v, index)
        elif isinstance(v, list):
            for i in range(len(v)):
                if(isinstance(v[i], float)):
                    v[i] = l[index]
                    index += 1
        elif isinstance(v, float):
            d[k] = l[index]
            index += 1
    return d