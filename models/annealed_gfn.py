import torch
import torch.nn as nn

from omegaconf import DictConfig

from typing import Optional, Any

from energy import BaseEnergy, AnnealedDensities

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import LearnedDiffusionConditional


class AnnealedGFN(SamplerModel):
    """
    GFN that learns annealed density path.

    It can be thought as GFN that can have backward policy with any parametrization.
    """

    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_conditional: Any,  # partial instance of DiffusionConditional
        backward_conditional: Any,  # partial instance of DiffusionConditional
        flow_model: Optional[torch.nn.Module] = None,
        device=torch.device("cuda"),
    ):
        super(AnnealedGFN, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            trajectory_length=trajectory_length,
            device=device,
            backprop_through_state=False,
        )

        self.annealed_energy = AnnealedDensities(
            energy_function=energy_function, prior_energy=prior_energy
        )

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        self.forward_conditional: LearnedDiffusionConditional = forward_conditional(
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            score_fn=self.energy_function.score,
        )

        self.backward_conditional: LearnedDiffusionConditional = backward_conditional(
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            score_fn=self.energy_function.score,
        )

        # TODO: Refactor this part later.
        self.forward_model = self.forward_conditional.joint_policy
        self.backward_model = self.backward_conditional.joint_policy
        self.fwd_lgv_scaler = self.forward_conditional.langevin_scaler
        self.bwd_lgv_scaler = self.backward_conditional.langevin_scaler

        # Flow model
        self.conditional_flow_model = flow_model

        # If flow is not learned, at least learn log Z (= total flow F(s_0))
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        # log Z ratio estimator
        self.logZ_ratio = torch.nn.Parameter(
            torch.zeros(trajectory_length, device=device)
        )

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

        return flow

    @property
    def learned_logZ(self):
        if self.conditional_flow_model is not None and self.is_dirac_prior:
            state = torch.zeros(self.sample_dim, device=self.device)
            time = 0.0

            encoded_state = self.state_encoder(state)
            encoded_time = self.time_encoder(time)

            flow = self.conditional_flow_model(encoded_state, encoded_time)
            return flow
        else:
            return self.logZ

    def get_logZ_ratio(self):
        return self.logZ_ratio

    def get_optimizer(self, optimizer_cfg: DictConfig):
        param_groups = [
            {"params": self.time_encoder.parameters()},
            {"params": self.state_encoder.parameters()},
            {"params": self.forward_conditional.joint_policy.parameters()},
            {"params": self.backward_conditional.joint_policy.parameters()},
        ]

        if self.forward_conditional.langevin_parametrization:
            param_groups += [
                {"params": self.forward_conditional.langevin_scaler.parameters()}
            ]

        if self.backward_conditional.langevin_parametrization:
            param_groups += [
                {"params": self.backward_conditional.langevin_scaler.parameters()}
            ]

        if self.conditional_flow_model is not None:
            param_groups += [
                {
                    "params": self.conditional_flow_model.parameters(),
                    "lr": optimizer_cfg.lr_flow,
                }
            ]

        param_groups += [{"params": self.logZ, "lr": optimizer_cfg.lr_flow}]
        param_groups += [{"params": self.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

        if optimizer_cfg.use_weight_decay:
            optimizer = torch.optim.Adam(
                param_groups,
                optimizer_cfg.lr_policy,
                weight_decay=optimizer_cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(param_groups, optimizer_cfg.lr_policy)

        return optimizer
