import torch
import constants as c
import numpy as np
from scipy import stats


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
    return 1/2+1/2*(np.tanh(KappaEQ*np.sin(2*np.pi*x)))