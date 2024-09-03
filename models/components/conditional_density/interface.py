import abc
import math

import torch
from torch import Tensor

import numpy as np


log_two_pi = math.log(2 * math.pi)


def gaussian_log_prob(x, mean, logvar):
    noise = (x - mean) / logvar.exp().sqrt()
    return (-0.5 * (log_two_pi + logvar + noise**2)).sum(-1)


class ConditionalDensity(abc.ABC):
    """
    Abstract class for Conditional density family.
    Here we assume gaussian as the default.
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
    ) -> Tensor:
        mean, logvar = param["mean"], param["logvar"]
        sample = mean + (logvar / 2).exp() * torch.randn_like(mean)
        return sample

    @staticmethod
    def add_to_std_in_param_dict(
        additional_std: float,
        param: dict,
    ):
        mean, logvar = param["mean"], param["logvar"]

        if additional_std <= 0.0:
            # For weired value, we don't do anything.
            new_logvar = logvar
        else:
            log_additional_var = torch.full_like(logvar, np.log(additional_std) * 2)
            new_logvar = torch.logaddexp(logvar, log_additional_var)

        return {
            "mean": mean,
            "logvar": new_logvar,
        }

    @staticmethod
    def split_gaussian_params(tensor):
        mean, logvar = torch.chunk(tensor, 2, dim=-1)
        return mean, logvar
