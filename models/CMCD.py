import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

from energy import BaseEnergy, AnnealedDensities

from .base_model import SamplerModel

from .components.architectures import *

log_two_pi = math.log(2 * math.pi)


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def gaussian_log_prob(x, mean, logvar):
    noise = (x - mean) / logvar.exp().sqrt()
    return (-0.5 * (log_two_pi + logvar + noise**2)).sum(1)


class CMCDSampler(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        control_model: nn.Module,
        base_diffusion_rate: float = 1.0,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        device=torch.device("cuda"),
    ):
        super(CMCDSampler, self).__init__(
            energy_function=energy_function,
            trajectory_length=trajectory_length,
            device=device,
        )

        self.prior_energy = prior_energy

        self.annealed_energy = AnnealedDensities(
            energy_function=energy_function, prior_energy=prior_energy
        )

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        self.control_model = control_model

        # log Z ratio estimator (This one only used in annealed-GFN)
        self.logZ_ratio = torch.nn.Parameter(
            torch.zeros(trajectory_length, device=device)
        )

        # Clipping on model output
        self.clipping: bool = clipping
        self.lgv_clip: float = lgv_clip
        self.gfn_clip: float = gfn_clip

        self.learned_variance = False

        # base diffusion rate, fixed sigma
        self.base_diffusion_rate = base_diffusion_rate

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)

        # mask the model output and set the variance as constant.
        logvar = torch.full_like(
            logvar, np.log(self.base_diffusion_rate) * 2.0 + np.log(2)
        )

        return mean, logvar

    def get_score_and_control(self, state: torch.Tensor, time: torch.Tensor):
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        encoded_state = self.state_encoder(state)

        mean_and_logvar = self.control_model(encoded_state, encoded_time)

        # Get annealed score and clip it.
        annealed_score = self.annealed_energy.score(time, state)
        if self.clipping:
            annealed_score = torch.clip(annealed_score, -self.lgv_clip, self.lgv_clip)

        return mean_and_logvar, annealed_score

    def get_forward_params(self, state: torch.Tensor, time: float):
        """
        Get forward conditional density's parameters.
        For given state s and time t, return the parameters of p_F(-| s_t).
        """
        mean_and_logvar, annealed_score = self.get_score_and_control(state, time)

        # Add annealed score to the control variate.
        mean_and_logvar[..., : self.sample_dim] += (
            self.base_diffusion_rate**2
        ) * annealed_score

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = self.split_params(mean_and_logvar)
        return {"mean": mean, "logvar": logvar}

    def get_backward_params(self, state: torch.Tensor, time: float):
        """
        Get backward conditional density's parameters.
        For given state s and time t, return the parameters of p_B(-| s_t).
        """
        mean_and_logvar, annealed_score = self.get_score_and_control(state, time)

        # Subtract annealed score to the control variate.
        mean_and_logvar[..., : self.sample_dim] -= (
            self.base_diffusion_rate**2
        ) * annealed_score

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = self.split_params(mean_and_logvar)

        return {"mean": mean, "logvar": logvar}

    def get_next_state(
        self,
        state: torch.Tensor,
        time: float,
        exploration_schedule: Optional[Callable[[float], float]] = None,
        stochastic_backprop: bool = False,
    ):
        """
        For given state s and time t,
        sample next state s_{t+1} from p_F(-| s_t) and
        return the parameters of p_F(-| s_t).

        Note that sample can be generated via additional exploration,
        and follows little different distribution from p_F(-| s_t).
        """
        pf_params = self.get_forward_params(state, time)
        pf_mean, pf_logvar = pf_params["mean"], pf_params["logvar"]

        if exploration_schedule is not None:
            exploration_std = exploration_schedule(time)
            pf_logvar_for_sample = self.add_more_exploration(pf_logvar, exploration_std)
        else:
            pf_logvar_for_sample = pf_logvar

        if stochastic_backprop:
            pf_mean_for_sample = pf_mean
        else:
            pf_mean_for_sample = pf_mean.detach()

        next_state = (
            state
            + self.dt * pf_mean_for_sample
            + np.sqrt(self.dt)
            * (pf_logvar_for_sample / 2).exp()
            * torch.randn_like(state, device=self.device)
        )

        return next_state, pf_params

    def get_prev_state(
        self,
        state: torch.Tensor,
        time: float,
        stochastic_backprop: bool = False,
    ):
        """
        For given state s and time t,
        sample next state s_{t-1} from p_B(-| s_t) and
        return the parameters of p_B(-| s_t).

        Note that sample can be generated via additional exploration,
        and follows little different distribution from p_B(-| s_t).
        """
        pb_params = self.get_backward_params(state, time)
        pb_mean, pb_logvar = pb_params["mean"], pb_params["logvar"]

        if stochastic_backprop:
            pb_mean_for_sample = pb_mean
        else:
            pb_mean_for_sample = pb_mean.detach()

        prev_state = (
            state
            + self.dt * pb_mean_for_sample
            + np.sqrt(self.dt)
            * (pb_logvar / 2).exp()
            * torch.randn_like(state, device=self.device)
        )

        return prev_state, pb_params

    def get_forward_logprob(
        self,
        next: torch.Tensor,
        cur: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        pf_mean, pf_logvar = params["mean"], params["logvar"]
        return gaussian_log_prob(
            next, cur + self.dt * pf_mean, np.log(self.dt) + pf_logvar
        )

    def get_backward_logprob(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        pb_mean, pb_logvar = params["mean"], params["logvar"]
        return gaussian_log_prob(
            cur, prev + self.dt * pb_mean, np.log(self.dt) + pb_logvar
        )

    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        return self.prior_energy.sample(batch_size, device=self.device)

    def get_logprob_initial_state(self, init_state: torch.Tensor) -> torch.Tensor:
        return self.prior_energy.log_prob(init_state)

    def add_more_exploration(self, log_var: torch.Tensor, exploration_std: float):
        if exploration_std <= 0.0:
            # For weired value of exploration_std, we don't add exploration noise.
            # But why detach...?
            pflogvars_sample = log_var.detach()
        else:
            log_additional_var = torch.full_like(
                log_var, np.log(exploration_std / np.sqrt(self.dt)) * 2
            )
            pflogvars_sample = torch.logaddexp(log_var, log_additional_var)

        return pflogvars_sample

    def sleep_phase_sample(self, batch_size, exploration_std):
        initial_states = self.generate_initial_state(batch_size)

        trajectories, _ = self.get_forward_trajectory(initial_states, exploration_std)

        return trajectories[:, -1]

    def forward(self, state, exploration_std=None, log_r=None):
        return self.get_forward_trajectory(state, exploration_std, log_r)

    def get_logZ_ratio(self):
        return self.logZ_ratio
