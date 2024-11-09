import torch
from typing import Optional

import torch.nn as nn

from omegaconf import DictConfig

from task import BaseEnergy, AnnealedDensities

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import ControlledMCConditional, MCConditional


class AISSampler(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        control_model: nn.Module,
        flow_model: Optional[torch.nn.Module] = None,
        base_std: float = 1.0,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        device=torch.device("cuda"),
    ):
        super(AISSampler, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            optimizer_cfg=optimizer_cfg,
            trajectory_length=trajectory_length,
            device=device,
        )

        self.annealed_energy = AnnealedDensities(
            energy_function=energy_function, prior_energy=prior_energy
        )

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        self.control_model = control_model

        self.forward_conditional = ControlledMCConditional(
            dt=self.dt,
            sample_dim=self.sample_dim,
            state_encoder=self.state_encoder,
            time_encoder=self.time_encoder,
            control_model=self.control_model,
            do_control_plus_score=True,
            annealed_score_fn=self.annealed_energy.score,
            base_std=base_std,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
        )

        self.backward_conditional = MCConditional(
            dt=self.dt,
            sample_dim=self.sample_dim,
            annealed_score_fn=self.annealed_energy.score,
            base_std=base_std,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
        )

        # Conditional flow model f(s, t) (0 <= t < 1)
        self.conditional_flow_model = flow_model

        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

    def get_flow_from_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.conditional_flow_model is None:
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

        flow = self.conditional_flow_model(encoded_trajectory, encoded_time).squeeze(-1)
        energy = self.annealed_energy.energy(time.squeeze(-1), trajectory)

        return flow - energy

    @property
    def learned_logZ(self):
        if self.conditional_flow_model is not None and self.is_dirac_prior:
            state = torch.zeros(self.sample_dim, device=self.device)
            time = 0.0

            encoded_state = self.state_encoder(state)
            encoded_time = self.time_encoder(time)

            flow = self.conditional_flow_model(encoded_state, encoded_time).squeeze(-1)
            return flow
        else:
            return self.logZ

    def param_groups(self):
        optimizer_cfg = self.optimizer_cfg

        param_groups = [
            {"params": self.time_encoder.parameters()},
            {"params": self.state_encoder.parameters()},
            {"params": self.control_model.parameters()},
        ]

        if self.conditional_flow_model is not None:
            param_groups += [
                {
                    "params": self.conditional_flow_model.parameters(),
                    "lr": optimizer_cfg.lr_flow,
                }
            ]

        param_groups += [{"params": self.logZ, "lr": 0.1}]

        return param_groups

    @property
    def learned_logZ(self):
        return self.logZ
