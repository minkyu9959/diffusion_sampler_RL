import torch
import torch.nn as nn

from omegaconf import DictConfig

from typing import Optional, Any

from energy import BaseEnergy, AnnealedDensities, DiracDeltaEnergy

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
        optimizer_cfg: DictConfig,
        trajectory_length: int,
        state_encoder: nn.Module,
        time_encoder: nn.Module,
        forward_policy: nn.Module,
        backward_policy: nn.Module,
        langevin_scaler: Any = None,  # partial instance of langevin scaler
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
        base_std: float = 1.0,
        flow_model: Optional[torch.nn.Module] = None,
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

        self.forward_conditional = LearnedDiffusionConditional(
            sample_dim=self.sample_dim,
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            joint_policy=self.forward_policy,
            langevin_scaler=self.forward_langevin_scaler,
            score_fn=self.energy_function.score,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
            learn_variance=learn_variance,
            log_var_range=log_var_range,
            base_std=base_std,
        )

        self.backward_conditional = LearnedDiffusionConditional(
            sample_dim=self.sample_dim,
            dt=self.dt,
            state_encoder=state_encoder,
            time_encoder=time_encoder,
            joint_policy=self.backward_policy,
            langevin_scaler=self.backward_langevin_scaler,
            score_fn=self.energy_function.score,
            clipping=clipping,
            lgv_clip=lgv_clip,
            gfn_clip=gfn_clip,
            learn_variance=learn_variance,
            log_var_range=log_var_range,
            base_std=base_std,
        )

        # Flow model
        self.conditional_flow_model = flow_model

        # learn log Z (= total flow F(s_0))
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        if fixed_logZ_ratio:
            logZ_ratio = self.annealed_energy.logZ_ratios(10000, trajectory_length)
            self.logZ_ratio = torch.nn.Parameter(logZ_ratio, requires_grad=False)
        else:
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
            # If we use dirac prior and we have flow model,
            # we can calculate log Z as flow(x, 0).
            state = torch.zeros(self.sample_dim, device=self.device)
            time = 0.0

            encoded_state = self.state_encoder(state)
            encoded_time = self.time_encoder(time)

            flow = self.conditional_flow_model(encoded_state, encoded_time)
            return flow
        else:
            # Else, flow(s) = log Z =/ flow(x, 0), so we return learned log Z.
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

        if self.conditional_flow_model is not None:
            param_groups += [
                {
                    "params": self.conditional_flow_model.parameters(),
                    "lr": optimizer_cfg.lr_flow,
                }
            ]

        param_groups += [{"params": self.logZ, "lr": optimizer_cfg.lr_flow}]
        param_groups += [{"params": self.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

        return param_groups
