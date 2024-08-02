"""
Here we implement the abstract base class for sequential sampler models
which use both forward, backward path measure.
"""

"""
    Trajectories Tensor Layout and Iterator on them.

                                    <---- Time ---->
    -------------------------------------------------------------------------------
    |                |---------|       |---------|       |---------|              |
    |       ...      |   prev  |  <->  |   cur   |  <->  |   next  |     ...      |
    |                |---------|       |---------|       |---------|              |
    -------------------------------------------------------------------------------
                          ^                 ^                 ^
                       prev_idx          cur_idx           next_idx
                       prev_time         cur_time          next_time

    Note:
    - SamplerModel._forward_iter iteratively yields (cur, next) from start to end.

    - SamplerModel._backward_iter iteratively yields (prev, cur) from end to start.
"""

from typing import Optional, Callable

import torch

from omegaconf import DictConfig

from .components.conditional_density import ConditionalDensity

from energy import BaseEnergy


class SamplerModel(torch.nn.Module):
    """
    Abstract base class for sequential sampler models
    which use both forward, backward path measure.

    We assume that both forward, backward conditional density is Gaussian.
    """

    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        trajectory_length: int = 100,
        device=torch.device("cuda"),
        backprop_through_state: bool = False,
    ):
        super(SamplerModel, self).__init__()

        self.prior_energy = prior_energy
        self.energy_function = energy_function
        self.optimizer_cfg = optimizer_cfg

        self.sample_dim = energy_function.data_ndim
        self.trajectory_length = trajectory_length

        self.dt = 1.0 / trajectory_length

        self.backprop_through_state = backprop_through_state

        self.device = device

    def set_conditional_density(
        self,
        forward_conditional: ConditionalDensity,
        backward_conditional: ConditionalDensity,
    ):
        self.forward_conditional = forward_conditional
        self.backward_conditional = backward_conditional

    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        """
        Generate initial state using prior energy function.
        """
        return self.prior_energy.sample(batch_size, device=self.device)

    def get_logprob_initial_state(self, init_state: torch.Tensor) -> torch.Tensor:
        """
        Return the log probability of initial states.
        """
        # Prior energy must support the log_prob method.
        return self.prior_energy.log_prob(init_state)

    def get_forward_trajectory(
        self,
        initial_states: torch.Tensor,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate forward trajectory from given initial states.
        This function returns trajectory and its probability (both forward and backward).

        kwargs is sent down to internal methods.

        Returns:
            trajectories:
            torch.tensor with shape (batch_size, trajectory_length + 1, sample_dim)

            log_forward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)

            log_backward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)
        """
        assert initial_states.dim() == 2 and initial_states.shape[1] == self.sample_dim

        batch_size = initial_states.shape[0]

        # Create empty (zero) tensor with corresponding size.
        logpf, logpb, trajectories = self._allocate_memory(batch_size)

        trajectories[:, 0] = cur_state = initial_states

        for cur_time, next_time, cur_idx, next_idx in self._forward_iter():
            # Get parameters of p_F(-| x_t).
            pf_params = self.forward_conditional.params(cur_state, cur_time)

            exploration_std = (
                exploration_schedule(cur_time) if exploration_schedule else 0.0
            )

            # Sample x_{t+1} ~ p_F(-| x_t) using parameters.
            next_state = self.forward_conditional.sample(
                pf_params, exploration_std=exploration_std
            )
            if not self.backprop_through_state:
                next_state = next_state.detach()

            trajectories[:, next_idx] = next_state

            logpf[:, cur_idx] = self.forward_conditional.log_prob(next_state, pf_params)

            # Get parameters of p_B(-|x_{t+1}).
            pb_params = self.backward_conditional.params(next_state, next_time)

            # Get probability of p_B(x_t|x_{t+1})
            logpb[:, cur_idx] = self.backward_conditional.log_prob(cur_state, pb_params)

            # This step is essential for back prop to work properly.
            # Reading cur_state from trajectory tensor will cause error.
            cur_state = next_state

        return trajectories, logpf, logpb

    def get_backward_trajectory(
        self,
        final_states: torch.Tensor,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate backward trajectory from given final states.
        This function returns trajectory and its probability (both forward and backward).

        Returns:
            trajectories:
            torch.tensor with shape (batch_size, trajectory_length + 1, sample_dim)

            log_forward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)

            log_backward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)
        """
        assert final_states.dim() == 2 and final_states.shape[1] == self.sample_dim

        batch_size = final_states.shape[0]

        # Create empty (zero) tensor with corresponding size.
        logpf, logpb, trajectories = self._allocate_memory(batch_size)

        trajectories[:, -1] = cur_state = final_states

        for cur_time, prev_time, cur_idx, prev_idx in self._backward_iter():
            # Get parameters of p_B(-| x_t).
            pb_params = self.backward_conditional.params(cur_state, cur_time)

            exploration_std = (
                exploration_schedule(cur_time) if exploration_schedule else 0.0
            )

            # Sample x_{t-1} ~ p_B(-| x_t) using parameters.
            prev_state = self.backward_conditional.sample(
                pb_params, exploration_std=exploration_std
            )
            if not self.backprop_through_state:
                prev_state = prev_state.detach()

            trajectories[:, prev_idx] = prev_state
            logpb[:, prev_idx] = self.backward_conditional.log_prob(
                prev_state, pb_params
            )

            # Get parameters of p_F(-| x_{t-1}).
            pf_params = self.forward_conditional.params(prev_state, prev_time)

            # Get probability of p_F(x_t| x_{t-1}).
            logpf[:, prev_idx] = self.forward_conditional.log_prob(cur_state, pf_params)

            # This step is essential for back prop to work properly.
            # Reading cur_state from trajectory tensor will cause error.
            cur_state = prev_state

        return trajectories, logpf, logpb

    def sample(self, batch_size: int) -> torch.Tensor:
        initial_states = self.generate_initial_state(batch_size)
        trajectories, _, _ = self.get_forward_trajectory(initial_states)
        return trajectories[:, -1]

    def sleep_phase_sample(
        self,
        batch_size: int,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ) -> torch.Tensor:
        initial_states = self.generate_initial_state(batch_size)

        trajectories, _, _ = self.get_forward_trajectory(
            initial_states, exploration_schedule
        )

        return trajectories[:, -1]

    def forward(
        self,
        state: torch.Tensor,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ):
        """
        Dummy forward function. Do not use this.
        """
        return self.get_forward_trajectory(state, exploration_schedule)

    def _allocate_memory(self, batch_size: int):
        """
        Allocate memory (i.e., create empty tensor) for trajectory generation.
        """
        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)

        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)

        trajectories = torch.zeros(
            (batch_size, self.trajectory_length + 1, self.sample_dim),
            device=self.device,
        )

        return logpf, logpb, trajectories

    def _forward_iter(self):
        for cur_idx in range(self.trajectory_length):
            next_idx = cur_idx + 1
            cur_time = cur_idx * self.dt
            next_time = next_idx * self.dt

            yield (cur_time, next_time, cur_idx, next_idx)

    def _backward_iter(self):
        for cur_idx in range(self.trajectory_length, 0, -1):
            prev_idx = cur_idx - 1
            cur_time = cur_idx * self.dt
            prev_time = prev_idx * self.dt

            yield (cur_time, prev_time, cur_idx, prev_idx)

    def get_optimizer(self):
        raise NotImplementedError("get_optimizer method must be implemented.")
