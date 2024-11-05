import torch

from omegaconf import DictConfig

from typing import Optional, Any, Callable

from task import BaseEnergy, AnnealedDensities, AnnealedEnergy

from .base_model import SamplerModel

from .components.architectures import *


class DoubleGFN(SamplerModel):
    """
    GFN that learns annealed density path.
    But with two copy targeting different time intervals.
    """

    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        first_gfn: Any,  # partial instance of SamplerModel
        second_gfn: Any,  # partial instance of SamplerModel
        trajectory_length: int,
        device=torch.device("cuda"),
    ):
        assert trajectory_length % 2 == 0

        super(DoubleGFN, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            optimizer_cfg=optimizer_cfg,
            trajectory_length=trajectory_length,
            device=device,
            backprop_through_state=False,
        )

        self.annealed_energy = AnnealedDensities(
            energy_function=energy_function, prior_energy=prior_energy
        )

        self.intermediate_energy = AnnealedEnergy(self.annealed_energy, 0.5)

        self.first_gfn: SamplerModel = first_gfn(
            prior_energy=prior_energy,
            energy_function=self.intermediate_energy,
            trajectory_length=trajectory_length // 2,
        )
        self.second_gfn: SamplerModel = second_gfn(
            prior_energy=self.intermediate_energy,
            energy_function=energy_function,
            trajectory_length=trajectory_length // 2,
        )

        # learn log Z (= total flow F(s_0))
        self.logZ = self.second_gfn.logZ
        self.intermediate_logZ = self.first_gfn.logZ

        self.second_gfn.generate_initial_state = self.get_intermediate_state
        self.second_gfn.get_logprob_initial_state = self.get_logprob_intermediate

    def get_intermediate_state(self, batch_size: int) -> torch.Tensor:
        return self.first_gfn.sample(batch_size)

    def get_logprob_intermediate(self, states: torch.Tensor) -> torch.Tensor:
        return -self.intermediate_energy.energy(states) - self.intermediate_logZ

    def get_forward_trajectory(
        self,
        initial_states: torch.Tensor,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ):
        traj1, logpf1, logpb1 = self.first_gfn.get_forward_trajectory(
            initial_states, exploration_schedule
        )

        traj2, logpf2, logpb2 = self.second_gfn.get_forward_trajectory(
            traj1[:, -1], exploration_schedule
        )

        # Concat traj1, traj2, logpf1, logpf2, logpb1, logpb2.
        traj = torch.cat((traj1, traj2), dim=1)
        logpf = torch.cat((logpf1, logpf2), dim=1)
        logpb = torch.cat((logpb1, logpb2), dim=1)

        return traj, logpf, logpb

    def get_backward_trajectory(
        self,
        final_states: torch.Tensor,
        exploration_schedule: Optional[Callable[[float], float]] = None,
    ):
        traj2, logpf2, logpb2 = self.second_gfn.get_backward_trajectory(
            final_states, exploration_schedule
        )

        traj1, logpf1, logpb1 = self.first_gfn.get_backward_trajectory(
            traj2[:, 0], exploration_schedule
        )

        # Concat traj1, traj2, logpf1, logpf2, logpb1, logpb2.
        traj = torch.cat((traj1, traj2), dim=1)
        logpf = torch.cat((logpf1, logpf2), dim=1)
        logpb = torch.cat((logpb1, logpb2), dim=1)

        return traj, logpf, logpb

    @property
    def learned_logZ(self):
        return self.logZ

    def get_optimizer(self):
        # Double GFN is composed of two independent GFN and each trained separately.
        # Plz use self.first_gfn.get_optimizer() or self.second_gfn.get_optimizer() instead.
        return None
