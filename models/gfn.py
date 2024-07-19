import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

from energy import BaseEnergy

from .base_model import SamplerModel

from .components.architectures import *

log_two_pi = math.log(2 * math.pi)


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def gaussian_log_prob(x, mean, logvar):
    noise = (x - mean) / logvar.exp().sqrt()
    return (-0.5 * (log_two_pi + logvar + noise**2)).sum(1)


class GFN(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_model: nn.Module,
        backward_model: Optional[nn.Module] = None,
        langevin_scaler: Optional[torch.nn.Module] = None,
        flow_model: Optional[torch.nn.Module] = None,
        log_var_range: float = 4.0,
        t_scale: float = 1.0,
        learned_variance: bool = True,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        pb_scale_range: float = 1.0,
        device=torch.device("cuda"),
    ):
        super(GFN, self).__init__(
            energy_function=energy_function,
            trajectory_length=trajectory_length,
            device=device,
        )

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        # These two model predict the mean and variance of Gaussian density.
        self.forward_model = forward_model
        self.backward_model = backward_model

        # Langevin scaler for Langevin parametrization.
        self.langevin_scaler = langevin_scaler
        self.langevin_parametrization = langevin_scaler is not None
        self.langevin_scaler_is_PIS = (
            type(self.langevin_scaler) == LangevinScalingModelPIS
        )

        # Flow model
        self.flow_model = flow_model

        if flow_model is None:
            # If flow is not learned, at least learn log Z (= total flow F(s_0))
            self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        # log Z ratio estimator (This one only used in annealed-GFN)
        self.logZ_ratio = torch.nn.Parameter(
            torch.zeros(trajectory_length, device=device)
        )

        # Clipping on model output
        self.clipping: bool = clipping
        self.lgv_clip: float = lgv_clip
        self.gfn_clip: float = gfn_clip

        self.learned_variance = learned_variance

        # -log_var_range < (learned log var) < log_var_range
        self.log_var_range = log_var_range

        # 1 - pb_scale_range < (backward correction) < 1 + pb_scale_range
        self.pb_scale_range = pb_scale_range

        # coefficient for noise.
        self.pf_std_per_traj = np.sqrt(t_scale)

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            # mask the model output and set the variance as constant.
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range

        return mean, logvar + np.log(self.pf_std_per_traj) * 2.0

    def get_forward_params(self, state: torch.Tensor, time: float):
        """
        Get forward conditional density's parameters.
        For given state s and time t, return the parameters of p_F(-| s_t).
        """
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        encoded_state = self.state_encoder(state)

        mean_and_logvar = self.forward_model(encoded_state, encoded_time)

        # Langevin parametrization trick, scale mean of forward density.
        if self.langevin_parametrization:
            grad_log_reward = -self.energy_function.score(state)

            grad_log_reward = torch.nan_to_num(grad_log_reward)
            if self.clipping:
                grad_log_reward = torch.clip(
                    grad_log_reward, -self.lgv_clip, self.lgv_clip
                )

            # Ad-hoc implementation for now. Need to be refactored.
            if self.langevin_scaler_is_PIS:
                scale = self.langevin_scaler(time)
            else:
                scale = self.langevin_scaler(encoded_state, encoded_time)

            mean_and_logvar[..., : self.sample_dim] += scale * grad_log_reward

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = self.split_params(mean_and_logvar)
        return {"mean": mean, "logvar": logvar}

    def get_backward_params(self, state: torch.Tensor, time: float):
        """
        Get backward conditional density's parameters.
        For given state s and time t, return the parameters of p_B(-| s_t).
        """
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]
        prev_time = time - self.dt

        if self.backward_model is not None:
            encoded_time = self.time_encoder(time).repeat(batch_size, 1)
            encoded_state = self.state_encoder(state)
            pbs = self.backward_model(encoded_state, encoded_time)

            dmean, dvar = gaussian_params(pbs)

            mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            var_correction = 1 + dvar.tanh() * self.pb_scale_range
        else:
            mean_correction = torch.ones_like(state)
            var_correction = torch.ones_like(state)

        # Learned model adjust Brownian motion by multiplying correction coefficient to params.
        mean = state - self.dt * state / (time) * mean_correction
        var = (self.pf_std_per_traj**2) * (prev_time / time) * self.dt * var_correction

        return {"mean": mean, "var": var}

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
            pf_logvar_for_sample = pf_logvar_for_sample
        else:
            pf_mean_for_sample = pf_mean.detach()
            pf_logvar_for_sample = pf_logvar_for_sample.detach()

        next_state = (
            state
            + self.dt * pf_mean_for_sample
            + np.sqrt(self.dt)
            * (pf_logvar_for_sample / 2).exp()
            * torch.randn_like(state, device=self.device)
        )

        return next_state, pf_params

    def get_prev_state(self, state: torch.Tensor, time: float, **kwargs):
        """
        For given state s and time t,
        sample next state s_{t-1} from p_B(-| s_t) and
        return the parameters of p_B(-| s_t).

        Note that sample can be generated via additional exploration,
        and follows little different distribution from p_B(-| s_t).
        """
        pb_params = self.get_backward_params(state, time)
        pb_mean, pb_var = pb_params["mean"], pb_params["var"]

        # For backward transition, we don't add exploration noise.
        # For backward transition, we don't propagate through sample.
        prev_state = pb_mean.detach() + pb_var.sqrt().detach() * torch.randn_like(
            state, device=self.device
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
        pb_mean, pb_var = params["mean"], params["var"]
        return gaussian_log_prob(prev, pb_mean, pb_var.log())

    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.sample_dim, device=self.device)

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

    def get_flow_from_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.flow_model is None:
            raise Exception("Flow model is not defined.")

        assert (
            trajectory.dim() == 3
            and trajectory.size(1) == self.trajectory_length + 1
            and trajectory.size(2) == self.sample_dim
        )

        batch_size = trajectory.shape[0]

        encoded_trajectory = self.state_encoder(trajectory)

        # time is (B, T + 1, 1) sized tensor
        time = (
            torch.linspace(0, 1, self.trajectory_length + 1, device=self.device)
            .repeat(batch_size, 1)
            .unsqueeze(-1)
        )

        # Now, encoded_time is (B, T + 1, t_emb_dim) sized tensor
        encoded_time = self.time_encoder(time)

        flow = self.flow_model(encoded_trajectory, encoded_time).squeeze(-1)

        return flow

    def get_learned_logZ(self, trajectory: torch.Tensor):
        if self.flow_model is None:
            return self.logZ
        else:
            flow = self.get_flow_from_trajectory(trajectory)
            return flow[:, 0]

    def get_logZ_ratio(self):
        return self.logZ_ratio
