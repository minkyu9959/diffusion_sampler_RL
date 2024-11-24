import torch
import math

import torch.nn as nn

from omegaconf import DictConfig

from task import BaseEnergy, AnnealedDensities

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import ControlledMCConditional, MCConditional


class CMCDSampler(SamplerModel):
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        control_model: nn.Module,
        base_std: float = 1.0,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        device=torch.device("cuda"),
    ):
        super(CMCDSampler, self).__init__(
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
        
        # self.backward_conditional = MCConditional(
        #     dt=self.dt,
        #     sample_dim=self.sample_dim,
        #     annealed_score_fn=self.annealed_energy.score,
        #     base_std=base_std,
        #     clipping=clipping,
        #     lgv_clip=lgv_clip,
        #     gfn_clip=gfn_clip,
        # )  

        self.backward_conditional = ControlledMCConditional(
            dt=self.dt,
            sample_dim=self.sample_dim,
            state_encoder=self.state_encoder,
            time_encoder=self.time_encoder,
            control_model=self.control_model,
            do_control_plus_score=False,
            annealed_score_fn=self.annealed_energy.score,
            base_std=base_std,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
        )


        # log Z ratio estimator (This one only used in annealed-GFN)
        self.logZ_ratio = torch.nn.Parameter(
            torch.zeros(trajectory_length, device=device)
        )

        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        self.annealing_schedule = None

    def get_logZ_ratio(self):
        return self.logZ_ratio

    def param_groups(self):
        optimizer_cfg = self.optimizer_cfg

        param_groups = [
            {"params": self.time_encoder.parameters()},
            {"params": self.state_encoder.parameters()},
            {"params": self.control_model.parameters()},
        ]

        param_groups += [{"params": self.logZ_ratio, "lr": optimizer_cfg.lr_flow}]
        return param_groups

    @property
    def learned_logZ(self):
        return self.logZ
