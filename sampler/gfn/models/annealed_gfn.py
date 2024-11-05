import torch
import torch.nn as nn

from omegaconf import DictConfig

from typing import Optional, Any

from task import BaseEnergy, AnnealedDensities, DiracDeltaEnergy

from .base_model import SamplerModel

from .components.architectures import *
from .components.conditional_density import (
    LearnedAnnealedDiffusionConditional,
    LearnedDiffusionConditional,
)


class AnnealedGFN(SamplerModel):
    """
    GFN that learns annealed density path.

    It can be thought as GFN that can have backward policy with any parametrization.
    """

    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
        optimizer_cfg: DictConfig,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_policy: nn.Module,
        backward_policy: nn.Module,
        annealing_step: int = 100,
        langevin_scaler: Any = None,  # partial instance of langevin scaler
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
        base_std: float = 1.0,
        annealing_schedule: Optional[torch.nn.Module] = None,
        device=torch.device("cuda"),
        fixed_logZ_ratio: bool = False,
    ):
        super(AnnealedGFN, self).__init__(
            energy_function=energy_function,
            prior_energy=prior_energy,
            optimizer_cfg=optimizer_cfg,
            trajectory_length=trajectory_length,
            device=device,
            backprop_through_state=False,
        )

        self.is_dirac_prior = type(prior_energy) == DiracDeltaEnergy

        self.annealed_energy = AnnealedDensities(
            energy_function=energy_function, prior_energy=prior_energy
        )
        self.annealing_step = annealing_step

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

        self.langevin_parametrization = langevin_scaler is not None

        if self.langevin_parametrization:
            self.forward_langevin_scaler = langevin_scaler()
            self.backward_langevin_scaler = langevin_scaler()
        else:
            self.forward_langevin_scaler = None
            self.backward_langevin_scaler = None

        self.forward_conditional = LearnedAnnealedDiffusionConditional(
            sample_dim=self.sample_dim,
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            joint_policy=self.forward_policy,
            langevin_scaler=self.forward_langevin_scaler,
            score_fn=self.annealed_energy.score,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
            learn_variance=learn_variance,
            log_var_range=log_var_range,
            base_std=base_std,
        )

        self.backward_conditional = LearnedAnnealedDiffusionConditional(
            sample_dim=self.sample_dim,
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            joint_policy=self.backward_policy,
            langevin_scaler=self.backward_langevin_scaler,
            score_fn=self.annealed_energy.score,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
            learn_variance=learn_variance,
            log_var_range=log_var_range,
            base_std=base_std,
        )

        # learn log Z (= total flow F(s_0))
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        if fixed_logZ_ratio:
            logZ_ratio = self.annealed_energy.logZ_ratios(10000, annealing_step)
            self.logZ_ratio = torch.nn.Parameter(logZ_ratio, requires_grad=False)
        else:
            # log Z ratio estimator
            self.logZ_ratio = torch.nn.Parameter(
                torch.zeros(annealing_step, device=device)
            )

        self.annealing_schedule = annealing_schedule

    @property
    def learned_logZ(self):
        return self.logZ

    def get_logZ_ratio(self):
        return self.logZ_ratio

    def param_groups(self):
        optimizer_cfg: DictConfig = self.optimizer_cfg

        param_groups = [
            {"params": self.time_encoder.parameters()},
            {"params": self.state_encoder.parameters()},
            {"params": self.forward_policy.parameters()},
            {"params": self.backward_policy.parameters()},
        ]

        if self.langevin_parametrization:
            param_groups += [
                {"params": self.forward_langevin_scaler.parameters()},
                {"params": self.backward_langevin_scaler.parameters()},
            ]

        param_groups += [{"params": self.logZ, "lr": optimizer_cfg.lr_flow}]
        param_groups += [{"params": self.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

        if self.annealing_schedule is not None:
            param_groups += [
                {
                    "params": self.annealing_schedule.parameters(),
                    "lr": optimizer_cfg.lr_flow,
                }
            ]

        return param_groups
