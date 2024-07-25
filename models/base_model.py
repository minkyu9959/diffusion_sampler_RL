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

import abc
import torch

from energy import BaseEnergy


class SamplerModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for sequential sampler models
    which use both forward, backward path measure.

    We assume that both forward, backward conditional density is Gaussian.
    """

    def __init__(
        self,
        energy_function: BaseEnergy,
        trajectory_length: int = 100,
        device=torch.device("cuda"),
    ):
        super(SamplerModel, self).__init__()

        self.energy_function = energy_function
        self.sample_dim = energy_function.data_ndim
        self.trajectory_length = trajectory_length

        self.dt = 1.0 / trajectory_length

        self.device = device

    @abc.abstractmethod
    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_logprob_initial_state(self, init_state: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_forward_params(self, state: torch.Tensor, time: float, **kwargs):
        """
        Get forward conditional density's parameters.
        For given state s and time t, return the parameters of p_F(-| s_t).
        """
        pass

    @abc.abstractmethod
    def get_backward_params(self, state: torch.Tensor, time: float, **kwargs):
        """
        Get backward conditional density's parameters.
        For given state s and time t, return the parameters of p_B(-| s_t).
        """
        pass

    @abc.abstractmethod
    def get_next_state(self, state: torch.Tensor, time: float, **kwargs):
        """
        For given state s and time t,
        sample next state s_{t+1} from p_F(-| s_t) and
        return the parameters of p_F(-| s_t).

        Note that sample can be generated via additional exploration,
        and follows little different distribution from p_F(-| s_t).
        """
        pass

    @abc.abstractmethod
    def get_prev_state(self, state: torch.Tensor, time: float, **kwargs):
        """
        For given state s and time t,
        sample next state s_{t-1} from p_B(-| s_t) and
        return the parameters of p_B(-| s_t).

        Note that sample can be generated via additional exploration,
        and follows little different distribution from p_B(-| s_t).
        """
        pass

    @abc.abstractmethod
    def get_forward_logprob(
        self,
        next: torch.Tensor,
        cur: torch.Tensor,
        params: dict,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_backward_logprob(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        params: dict,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def get_forward_trajectory(
        self, initial_states: torch.Tensor, **kwargs
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
            next_state, pf_params = self.get_next_state(cur_state, cur_time, **kwargs)

            trajectories[:, next_idx] = next_state

            logpf[:, cur_idx] = self.get_forward_logprob(
                next_state, cur_state, pf_params
            )

            pb_params = self.get_backward_params(next_state, next_time)

            logpb[:, cur_idx] = self.get_backward_logprob(
                cur_state, next_state, pb_params
            )

            # This step is essential for back prop to work properly.
            # Reading cur_state from trajectory tensor will cause error.
            cur_state = next_state

        return trajectories, logpf, logpb

    def get_backward_trajectory(
        self, final_states: torch.Tensor, **kwargs
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

        logpf, logpb, trajectories = self._allocate_memory(batch_size)

        trajectories[:, -1] = cur_state = final_states

        for cur_time, prev_time, cur_idx, prev_idx in self._backward_iter():
            prev_state, pb_params = self.get_prev_state(cur_state, cur_time, **kwargs)

            logpb[:, prev_idx] = self.get_backward_logprob(
                prev_state, cur_state, pb_params
            )

            trajectories[:, prev_idx] = prev_state

            pf_params = self.get_forward_params(prev_state, prev_time)
            logpf[:, prev_idx] = self.get_forward_logprob(
                cur_state, prev_state, pf_params
            )

            cur_state = prev_state

        return trajectories, logpf, logpb

    def sample(self, batch_size: int) -> torch.Tensor:
        initial_states = self.generate_initial_state(batch_size)
        trajectories, _, _ = self.get_forward_trajectory(initial_states)
        return trajectories[:, -1]

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
