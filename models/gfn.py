import torch

import numpy as np
import torch.nn as nn

from typing import Optional

from omegaconf import DictConfig

from energy import BaseEnergy, DiracDeltaEnergy

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import (
    LearnedDiffusionConditional,
    BrownianConditional,
    CorrectedBrownianConditional,
)


class GFN(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_model: nn.Module,
        backward_model: Optional[nn.Module] = None,
        langevin_scaler: Optional[torch.nn.Module] = None,
        flow_model: Optional[torch.nn.Module] = None,
        log_var_range: float = 4.0,
        pb_scale_range: float = 1.0,
        t_scale: float = 1.0,
        learned_variance: bool = True,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        backprop_through_state: bool = False,
        device=torch.device("cuda"),
    ):
        super(GFN, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            trajectory_length=trajectory_length,
            device=device,
            backprop_through_state=backprop_through_state,
        )

        self.prior_energy = prior_energy

        self.is_dirac_prior = type(prior_energy) == DiracDeltaEnergy

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        self.forward_model = forward_model
        self.backward_model = backward_model
        self.langevin_scaler = langevin_scaler

        # These two model predict the mean and variance of Gaussian density.
        forward_conditional = LearnedDiffusionConditional(
            self.sample_dim,
            self.dt,
            state_encoder,
            time_encoder,
            forward_model,
            langevin_scaler=langevin_scaler,
            score_fn=self.energy_function.score,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
            learn_variance=learned_variance,
            log_var_range=log_var_range,
            base_std=np.sqrt(t_scale),
        )

        if backward_model is None:
            backward_conditional = BrownianConditional(self.dt, np.sqrt(t_scale))
        else:
            backward_conditional = CorrectedBrownianConditional(
                self.dt,
                state_encoder,
                time_encoder,
                backward_model,
                base_std=np.sqrt(t_scale),
                mean_var_range=pb_scale_range,
            )

        self.set_conditional_density(
            forward_conditional=forward_conditional,
            backward_conditional=backward_conditional,
        )

        # Conditional flow model f(s, t) (0 <= t < 1)
        self.conditional_flow_model = flow_model

        # Total flow Z_theta
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        # log Z ratio estimator (This one only used in annealed-GFN)
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
            {"params": self.forward_model.parameters()},
        ]

        if self.backward_model is not None:
            param_groups += [
                {
                    "params": self.backward_model.parameters(),
                    "lr": optimizer_cfg.lr_back,
                }
            ]

        if self.langevin_scaler is not None:
            param_groups += [{"params": self.langevin_scaler.parameters()}]

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
