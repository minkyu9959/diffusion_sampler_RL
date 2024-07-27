import torch
from torch import Tensor

import numpy as np
import math

from .interface import ConditionalDensity, gaussian_params, gaussian_log_prob
from ..architectures import *


class BrownianConditional(ConditionalDensity):
    """
    Discretized Backward brownian bridge.

    It represents the conditional p(y|x) such that,
    y = (1 - dt/t) * x + noise * ((1 - dt/t) * sqrt(dt) * (base diffusion rate)).

    This process is fixed and used to model p_B(t->t-dt).
    """

    def __init__(self, dt: float, base_std: float):
        # Base diffusion rate is sigma in the diffusion process.

        self.dt = dt
        self.base_std = base_std

    def params(self, state: Tensor, time: float) -> dict:
        if math.isclose(time, self.dt):
            # p_B(dt->0) is determinisitc.
            return {
                "mean": torch.zeros_like(state),
                "var": torch.zeros_like(state),
                "deterministic": True,
            }

        prev_time = time - self.dt

        mean = state - self.dt * state / (time)
        var = torch.full_like(
            state, ((prev_time / time) * (self.base_std**2) * self.dt)
        )

        return {"mean": mean, "var": var}

    def log_prob(self, state: Tensor, param: dict) -> Tensor:
        if param.get("deterministic"):
            # We do not check the state is zero even though p_B(dt->0) is dirac delta.
            return torch.zeros(state.shape[:-1], device=state.device)

        mean, var = param["mean"], param["var"]

        return gaussian_log_prob(state, mean, var.log())

    def sample(
        self,
        param: dict,
        exploration_std: float = 0.0,
        # Fixed Brownian motion, so we don't use exploration.
    ) -> Tensor:
        mean, var = param["mean"], param["var"]

        sample = mean + torch.randn_like(mean) * var.sqrt()
        return sample


class CorrectedBrownianConditional(BrownianConditional):
    """
    Discretized Backward brownian bridge with neural network correction.

    It represents the conditional p(y|x) such that,
    y = (1 - dt/t) * x + noise * ((1 - dt/t) * sqrt(dt) * (base diffusion rate)).

    This process is used to model p_B(t->t-dt).
    """

    def __init__(
        self,
        dt: float,
        state_encoder: torch.nn.Module,
        time_encoder: torch.nn.Module,
        joint_policy: torch.nn.Module,
        base_std: float = 1.0,
        mean_var_range: float = 1.0,
    ):
        super().__init__(dt, base_std)

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder
        self.joint_policy = joint_policy

        self.mean_var_range = mean_var_range

    def params(self, state: Tensor, time: float):
        if math.isclose(time, self.dt):
            # p_B(dt->0) is determinisitc.
            return {
                "mean": torch.zeros_like(state),
                "var": torch.zeros_like(state),
                "determinisitc": True,
            }

        batch_size = state.shape[0]

        encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        encoded_state = self.state_encoder(state)
        mean_and_var = self.joint_policy(encoded_state, encoded_time)

        dmean, dvar = gaussian_params(mean_and_var)

        mean_correction = 1 + dmean.tanh() * self.mean_var_range
        var_correction = 1 + dvar.tanh() * self.mean_var_range

        prev_time = time - self.dt

        mean = state - self.dt * (state / time) * mean_correction
        var = (prev_time / time) * (self.base_std**2) * var_correction

        return {"mean": mean, "var": var}
