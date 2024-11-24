from typing import Union, Optional

import torch
from torch import Tensor

import numpy as np

from .interface import ConditionalDensity
from ..architectures import *


class LearnedDiffusionConditional(ConditionalDensity):
    """
    Conditional density family describing the diffusion process.
    It represents the Gaussian conditional p(y|x) such that,
    y = x + dt * mean + noise * (sqrt(dt) * sigma * base diffusion rate)
    where mean and sigma are learned.

    Learned sigma must lie between (-log_var_range, log_var_range).

    This can be used to model p_F(t->t+dt) or p_B(t->t-dt).
    """

    def __init__(
        self,
        sample_dim: int,
        dt: float,
        state_encoder: torch.nn.Module,
        time_encoder: torch.nn.Module,
        joint_policy: torch.nn.Module,
        langevin_scaler: Optional[torch.nn.Module] = None,
        score_fn = None,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
        base_std: float = 1.0,
        # Base diffusion rate is sigma in the diffusion process.
    ):
        self.sample_dim = sample_dim
        self.dt = dt

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder
        self.joint_policy = joint_policy

        self.langevin_parametrization = langevin_scaler is not None

        if self.langevin_parametrization:
            self.langevin_scaler = langevin_scaler

            self.score_fn = score_fn

            self.langevin_scaler_is_PIS = (
                type(self.langevin_scaler) == LangevinScalingModelPIS
            )

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.learn_variance = learn_variance
        self.log_var_range = log_var_range
        self.base_std = base_std

    def get_langevin_score(self, state: Tensor):
        grad_log_reward = self.score_fn(state)
        grad_log_reward = torch.nan_to_num(grad_log_reward)

        if self.clipping:
            grad_log_reward = torch.clip(grad_log_reward, -self.lgv_clip, self.lgv_clip)
        return grad_log_reward

    def transform_logvar(self, logvar: Tensor):
        if not self.learn_variance:
            # If we don't learn variance, we set it to zero.
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range

        logvar += np.log(self.base_std) * 2.0
        return logvar

    def params(self, state: Tensor, time: Union[float, Tensor]) -> dict:
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        if type(time) is float:
            encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        else:
            encoded_time = self.time_encoder(time)

        encoded_state = self.state_encoder(state)

        mean_and_logvar = self.joint_policy(encoded_state, encoded_time)

        # Langevin parametrization trick, scale mean of forward density.
        if self.langevin_parametrization:
            score = self.get_langevin_score(state)

            if self.langevin_scaler_is_PIS:
                scale = self.langevin_scaler(time)
            else:
                scale = self.langevin_scaler(encoded_state, encoded_time)

            mean_and_logvar[..., : self.sample_dim] += scale * score

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = ConditionalDensity.split_gaussian_params(mean_and_logvar)

        logvar = self.transform_logvar(logvar)

        mean = mean * self.dt
        mean = mean + state

        # Multiply dt to the variance.
        logvar = logvar + np.log(self.dt)

        return {"mean": mean, "logvar": logvar}


class LearnedAnnealedDiffusionConditional(ConditionalDensity):
    """
    Conditional density family describing the diffusion process.
    It represents the Gaussian conditional p(y|x) such that,
    y = x + dt * mean + noise * (sqrt(dt) * sigma * base diffusion rate)
    where mean and sigma are learned.

    Learned sigma must lie between (-log_var_range, log_var_range).

    This can be used to model p_F(t->t+dt) or p_B(t->t-dt).
    """

    def __init__(
        self,
        sample_dim: int,
        dt: float,
        state_encoder: torch.nn.Module,
        time_encoder: torch.nn.Module,
        joint_policy: torch.nn.Module,
        langevin_scaler: Optional[torch.nn.Module] = None,
        score_fn=None,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
        base_std: float = 1.0,
        # Base diffusion rate is sigma in the diffusion process.
    ):
        self.sample_dim = sample_dim
        self.dt = dt

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder
        self.joint_policy = joint_policy

        self.langevin_parametrization = langevin_scaler is not None

        if self.langevin_parametrization:
            self.langevin_scaler = langevin_scaler

            self.score_fn = score_fn

            self.langevin_scaler_is_PIS = (
                type(self.langevin_scaler) == LangevinScalingModelPIS
            )

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.learn_variance = learn_variance
        self.log_var_range = log_var_range
        self.base_std = base_std

    def get_langevin_score(self, state: Tensor, time: float):
        grad_log_reward = self.score_fn(time, state)
        grad_log_reward = torch.nan_to_num(grad_log_reward)

        if self.clipping:
            grad_log_reward = torch.clip(grad_log_reward, -self.lgv_clip, self.lgv_clip)
        return grad_log_reward

    def transform_logvar(self, logvar: Tensor):
        if not self.learn_variance:
            # If we don't learn variance, we set it to zero.
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range

        logvar += np.log(self.base_std) * 2.0
        return logvar

    def params(self, state: Tensor, time: float) -> dict:
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        if type(time) is float:
            encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        else:
            encoded_time = self.time_encoder(time)

        encoded_state = self.state_encoder(state)

        mean_and_logvar = self.joint_policy(encoded_state, encoded_time)

        # Langevin parametrization trick, scale mean of forward density.
        if self.langevin_parametrization:
            score = self.get_langevin_score(state, time)

            if self.langevin_scaler_is_PIS:
                scale = self.langevin_scaler(time)
            else:
                scale = self.langevin_scaler(encoded_state, encoded_time)

            mean_and_logvar[..., : self.sample_dim] += scale * score

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = ConditionalDensity.split_gaussian_params(mean_and_logvar)

        logvar = self.transform_logvar(logvar)

        mean = mean * self.dt
        mean = mean + state

        # Multiply dt to the variance.
        logvar = logvar + np.log(self.dt)

        return {"mean": mean, "logvar": logvar}
