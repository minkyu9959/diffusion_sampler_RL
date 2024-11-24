import torch

import numpy as np
import torch.nn as nn

from typing import Optional

from omegaconf import DictConfig

from task import BaseEnergy, DiracDeltaEnergy

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import (
    LearnedDiffusionConditional,
    BrownianConditional,
    CorrectedBrownianConditional,
    ControlledMCConditional,
    BackwardMCConditional,
)
from task.energies.annealed_energy import AnnealedDensities


class GFN(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_policy: nn.Module,
        backward_policy: Optional[nn.Module] = None,
        langevin_scaler: Optional[torch.nn.Module] = None,
        flow_model: Optional[torch.nn.Module] = None,
        log_var_range: float = 4.0,
        pb_scale_range: float = 1.0,
        t_scale: float = 1.0,
        learned_variance: bool = True,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        fixed_logZ: bool = False,
        backprop_through_state: bool = False,
        device=torch.device("cuda"),
    ):
        super(GFN, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            optimizer_cfg=optimizer_cfg,
            trajectory_length=trajectory_length,
            device=device,
            backprop_through_state=backprop_through_state,
        )

        self.prior_energy = prior_energy
        self.annealed_energy = AnnealedDensities(energy_function, prior_energy)

        self.is_dirac_prior = type(prior_energy) == DiracDeltaEnergy

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder

        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.langevin_scaler = langevin_scaler

        if langevin_scaler is not None:
            self.langevin_parametrization = True

        # These two model predict the mean and variance of Gaussian density.
        
        # forward_conditional = LearnedDiffusionConditional(
        #     self.sample_dim,
        #     self.dt,
        #     state_encoder,
        #     time_encoder,
        #     forward_policy,
        #     langevin_scaler=langevin_scaler,
        #     score_fn=self.energy_function.score, 
        #     clipping=clipping,
        #     lgv_clip=lgv_clip,
        #     gfn_clip=gfn_clip,
        #     learn_variance=learned_variance,
        #     log_var_range=log_var_range,
        #     base_std=np.sqrt(t_scale),
        # )
        
        forward_conditional = ControlledMCConditional(
            dt=self.dt,
            sample_dim=self.sample_dim,
            state_encoder=self.state_encoder,
            time_encoder=self.time_encoder,
            control_model=self.forward_policy,
            do_control_plus_score=True,
            annealed_score_fn=self.annealed_energy.score,
            base_std=np.sqrt(t_scale),
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
        )
               
        backward_conditional = BackwardMCConditional(
            dt=self.dt,
            sample_dim=self.sample_dim,
            annealed_score_fn=self.annealed_energy.score,
            base_std=np.sqrt(t_scale),
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
        )       
                
        # if backward_policy is None:
        #     backward_conditional = BrownianConditional(self.dt, np.sqrt(t_scale))
            
        # else:
        #     backward_conditional = CorrectedBrownianConditional(
        #         self.dt,
        #         state_encoder,
        #         time_encoder,
        #         backward_policy,
        #         base_std=np.sqrt(t_scale),
        #         mean_var_range=pb_scale_range,
        #     )

        self.set_conditional_density(
            forward_conditional=forward_conditional,
            backward_conditional=backward_conditional,
        )

        # Conditional flow model f(s, t) (0 <= t < 1)
        self.conditional_flow_model = flow_model

        # Total flow Z_theta
        if fixed_logZ:
            logZ = self.energy_function.ground_truth_logZ
            self.logZ = torch.nn.Parameter(
                torch.tensor(logZ, device=self.device), requires_grad=False
            )
        else:
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

        return flow

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
            {"params": self.forward_policy.parameters()},
        ]

        if self.backward_policy is not None:
            param_groups += [
                {
                    "params": self.backward_policy.parameters(),
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
        return param_groups
