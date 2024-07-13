import abc
import torch

from energy import BaseEnergy


class SamplerModel(torch.nn.Module, metaclass=BaseEnergy):
    """
    Abstract base class for sampler models which use both forward, backward path measure.
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

        self.device = device

    @abc.abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_forward_trajectory(
        self, initial_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate forward trajectory from given initial states.
        This function returns trajectory and its probability (both forward and backward).

        Returns:
            states:
            torch.tensor with shape (batch_size, trajectory_length + 1, sample_dim)

            log_forward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)

            log_backward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)
        """
        pass

    @abc.abstractmethod
    def get_backward_trajectory(
        self, final_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate backward trajectory from given final states.
        This function returns trajectory and its probability (both forward and backward).

        Returns:
            states:
            torch.tensor with shape (batch_size, trajectory_length + 1, sample_dim)

            log_forward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)

            log_backward_conditional_probability:
            torch.tensor with shape (batch_size, trajectory_length)
        """
        pass

    @abc.abstractmethod
    def generate_initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.sample_dim).to(self.device)
