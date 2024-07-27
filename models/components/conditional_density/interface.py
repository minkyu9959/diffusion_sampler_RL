import abc
import math

import torch
from torch import Tensor

import numpy as np


log_two_pi = math.log(2 * math.pi)


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def gaussian_log_prob(x, mean, logvar):
    noise = (x - mean) / logvar.exp().sqrt()
    return (-0.5 * (log_two_pi + logvar + noise**2)).sum(1)


def add_exploration_to_logvar(exploration_std: float, log_var: torch.Tensor):
    if exploration_std <= 0.0:
        # For weired value of exploration_std, we don't add exploration noise.
        logvars_sample = log_var
    else:
        log_additional_var = torch.full_like(log_var, np.log(exploration_std) * 2)
        logvars_sample = torch.logaddexp(log_var, log_additional_var)
    return logvars_sample


class ConditionalDensity(abc.ABC):
    """
    Abstract class for Conditional density family.
    Here we assume density as gaussian.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def params(self, state: Tensor, time: float) -> dict:
        pass

    def log_prob(self, state: Tensor, param: dict) -> Tensor:
        return gaussian_log_prob(state, param["mean"], param["logvar"])

    def sample(
        self,
        param: dict,
        exploration_std: float = 0.0,
    ) -> Tensor:
        mean, logvar = param["mean"], param["logvar"]

        logvar = add_exploration_to_logvar(exploration_std, logvar)

        sample = mean + (logvar / 2).exp() * torch.randn_like(mean)
        return sample
