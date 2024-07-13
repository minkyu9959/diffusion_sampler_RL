import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

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
        self.dt = 1.0 / trajectory_length

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        # These two model predict the mean and variance of Gaussian density.
        self.forward_model = forward_model
        self.backward_model = backward_model

        # Langevin scaler for Langevin parametrization.
        self.langevin_scaler = langevin_scaler

        # Flow model
        self.flow_model = flow_model

        # Clipping on model output
        self.clipping: bool = clipping
        self.lgv_clip: float = lgv_clip
        self.gfn_clip: float = gfn_clip

        self.learned_variance = learned_variance

        # -log_var_range < (learned log var) < log_var_range
        self.log_var_range = log_var_range

        # 1 - pb_scale_range < (backward correction) < 1 + pb_scale_range
        self.pb_scale_range = pb_scale_range

        # Constant noise per trajectory.
        self.pf_std_per_traj = np.sqrt(t_scale)

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            # If you don't want to learn variance,
            # mask the model output and set the variance as constant.
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range

        return mean, logvar + np.log(self.pf_std_per_traj) * 2.0

    def predict_next_state(self, state: torch.Tensor, time: float):
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        encoded_state = self.state_encoder(state)

        # next_state is (batch_size, 2 * sample_dim) sized tensor.
        # Each chunk (batch_size, sample_dim), (batch_size, sample_dim) represents mean and log variance.
        mean_and_logvar = self.forward_model(encoded_state, encoded_time)

        if self.langevin_scaler is not None:
            grad_log_reward = -self.energy_function.score(state)

            grad_log_reward = torch.nan_to_num(grad_log_reward)
            if self.clipping:
                grad_log_reward = torch.clip(
                    grad_log_reward, -self.lgv_clip, self.lgv_clip
                )

            # Langevin parametrization trick.
            # Scaling mean of forward conditional density.
            scale = self.langevin_scaling_model(state, time)
            mean_and_logvar[..., : self.sample_dim] += scale * grad_log_reward

        if self.flow_model is not None:
            flow = self.flow_model(encoded_state, encoded_time).squeeze(-1).squeeze(-1)
        else:
            flow = None

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = self.split_params(mean_and_logvar)
        return mean, logvar, flow

    def predict_prev_state(self, state: torch.Tensor, time: float):
        batch_size = state.shape[0]
        prev_time = time - self.dt

        if self.backward_model is not None:
            encoded_time = self.time_encoder(time).repeat(batch_size, 1)
            encoded_state = self.state_encoder(state)
            pbs = self.backward_model(encoded_state, encoded_time)

            dmean, dvar = gaussian_params(pbs)

            back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
        else:
            back_mean_correction = torch.ones_like(state)
            back_var_correction = torch.ones_like(state)

        back_mean = state - self.dt * state / (time) * back_mean_correction

        back_var = (
            (self.pf_std_per_traj**2)
            * (prev_time / time)
            * self.dt
            * back_var_correction
        )

        return back_mean, back_var

    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.sample_dim, device=self.device)

    def allocate_memory(self, batch_size: int):
        """
        Allocate memory (i.e., create empty tensor) for trajectory generation.
        """
        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)

        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)

        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)

        trajectories = torch.zeros(
            (batch_size, self.trajectory_length + 1, self.sample_dim),
            device=self.device,
        )

        return logpf, logpb, logf, trajectories

    def add_more_exploration(self, log_var, exploration_std):
        if exploration_std is None:
            pflogvars_sample = log_var
        elif exploration_std <= 0.0:
            # For weired value of exploration_std, we don't add exploration noise.
            # But why detach...?
            pflogvars_sample = log_var.detach()
        else:
            log_additional_var = torch.full_like(
                log_var, np.log(exploration_std / np.sqrt(self.dt)) * 2
            )
            pflogvars_sample = torch.logaddexp(log_var, log_additional_var)

        return pflogvars_sample

    def forward_iter(self, trajectories):
        for cur_idx in range(self.trajectory_length):
            next_idx = cur_idx + 1

            cur_state = trajectories[:, cur_idx, :]

            cur_time = cur_idx * self.dt
            next_time = next_idx * self.dt

            yield (cur_state, cur_time, next_time, cur_idx, next_idx)

    def backward_iter(self, trajectories):
        for cur_idx in range(self.trajectory_length, 0, -1):
            prev_idx = cur_idx - 1

            cur_state = trajectories[:, cur_idx, :]

            cur_time = cur_idx * self.dt
            prev_time = prev_idx * self.dt

            yield (cur_state, cur_time, prev_time, cur_idx, prev_idx)

    def get_forward_trajectory(
        self,
        initial_states: torch.Tensor,
        exploration_std=None,
        stochastic_backprop: bool = False,
        return_log_flow: bool = False,
    ):
        batch_size = initial_states.shape[0]
        log_reward = self.energy_function.log_reward
        logpf, logpb, logf, trajectories = self.allocate_memory(batch_size)

        # Tensor shape check
        assert logpf.size() == (batch_size, self.trajectory_length)
        assert logpb.size() == (batch_size, self.trajectory_length)
        assert logf.size() == (batch_size, self.trajectory_length + 1)
        assert trajectories.size() == (
            batch_size,
            self.trajectory_length + 1,
            self.sample_dim,
        )

        # Fill the initial states on trajectory.
        trajectories[:, 0, :] = initial_states

        for cur_state, cur_time, next_time, cur_idx, next_idx in self.forward_iter(
            trajectories
        ):
            pf_mean, pf_logvar, flow = self.predict_next_state(
                cur_state, cur_time, log_reward
            )

            logf[:, cur_idx] = flow

            pflogvars_for_sample = self.add_more_exploration(
                pf_logvar,
                exploration_std(cur_idx) if exploration_std is not None else None,
            )

            # For stochastic back prop (e.g. PIS),
            # we allow backpropagation through the mean, variance used for sampling.
            if stochastic_backprop:
                pf_mean_for_sample = pf_mean
                pflogvars_for_sample = pflogvars_for_sample
            else:
                pf_mean_for_sample = pf_mean.detach()
                pflogvars_for_sample = pflogvars_for_sample.detach()

            # Sampling next state
            next_state = (
                cur_state
                + self.dt * pf_mean_for_sample
                + np.sqrt(self.dt)
                * (pflogvars_for_sample / 2).exp()
                * torch.randn_like(cur_state, device=self.device)
            )

            # For this calculation, we must allow backpropagation through the mean, variance.
            logpf[:, cur_idx] = gaussian_log_prob(
                next_state, cur_state + self.dt * pf_mean, np.log(self.dt) + pf_logvar
            )

            if cur_idx > 0:
                pb_mean, pb_var = self.predict_prev_state(next_state, next_time)
                logpb[:, cur_idx] = gaussian_log_prob(cur_state, pb_mean, pb_var.log())
            else:
                # p_B(dt -> 0) is deterministic, so need not be calculated.
                pass

            trajectories[:, next_idx] = next_state

        if return_log_flow:
            return trajectories, logpf, logpb, logf
        else:
            return trajectories, logpf, logpb

    def get_backward_trajectory(
        self, final_states: torch.Tensor, return_log_flow: bool = False
    ):
        batch_size = final_states.shape[0]
        logpf, logpb, logf, trajectories = self.allocate_memory(batch_size)

        # Fill final states on trajectory.
        trajectories[:, -1] = final_states

        for cur_state, cur_time, prev_time, cur_idx, prev_idx in self.backward_iter(
            trajectories
        ):
            if cur_idx > 1:
                pb_mean, pb_var = self.predict_prev_state(cur_state, cur_time)

                prev_state = (
                    pb_mean.detach()
                    + pb_var.sqrt().detach()
                    * torch.randn_like(cur_state, device=self.device)
                )

                logpb[:, prev_idx] = gaussian_log_prob(
                    prev_state, pb_mean, pb_var.log()
                )
            else:
                # For p_B(dt -> 0), backward transition is deterministic.
                prev_state = torch.zeros(batch_size, device=self.device)

            trajectories[:, prev_idx] = prev_state

            pf_mean, pf_logvar, flow = self.predict_next_state(prev_state, prev_time)
            logf[:, prev_idx] = flow
            logpf[:, prev_idx] = gaussian_log_prob(cur_state, pf_mean, pf_logvar)

        if return_log_flow:
            return trajectories, logpf, logpb, logf
        else:
            return trajectories, logpf, logpb

    def sample(self, batch_size: int):
        initial_states = self.generate_initial_state(batch_size)

        trajectories, _ = self.get_forward_trajectory(initial_states, None)

        return trajectories[:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std):
        initial_states = self.generate_initial_state(batch_size)

        trajectories, _ = self.get_forward_trajectory(initial_states, exploration_std)

        return trajectories[:, -1]

    def forward(self, state, exploration_std=None, log_r=None):
        return self.get_forward_trajectory(state, exploration_std, log_r)
