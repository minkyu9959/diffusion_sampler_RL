from hydra import compose, initialize

import torch

from typing import Optional

from train import set_seed
from models import *
from energy import *


def load_energy_model_and_config(
    energy_name: str = "GMM25",
    model_name: str = "GFN-PIS",
    overrides: Optional[list] = None,
):
    if overrides is None:
        overrides = []

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="test.yaml",
            overrides=[
                f"energy={energy_name}",
                f"model={model_name}",
                *overrides,
            ],
        )

    set_seed(cfg.seed)

    energy = get_energy_function(cfg)

    model = get_model(cfg, energy).to(cfg.device)

    return energy, model, cfg


def test_stochastic_backprop():
    """
    Test stochastic back propagation by simulating one step of get_forward_trajectory loop.
    """

    energy, model, cfg = load_energy_model_and_config(
        model_name="GFN-PIS",
        overrides=[
            "model.backprop_through_state=true",
            "model.forward_conditional.joint_policy.zero_init=false",
        ],
    )

    state = init_states = torch.ones(
        (1, energy.ndim), device=cfg.device, requires_grad=True
    )

    # Allocate memory for logpf.
    logpf = torch.zeros((1, 2), device=cfg.device)

    # First step
    pf_params = model.forward_conditional.params(state, 0.5)
    next_state = model.forward_conditional.sample(pf_params)
    logpf[:, 0] = model.forward_conditional.log_prob(next_state, pf_params)

    # Second step
    state = next_state

    pf_params = model.forward_conditional.params(state, 0.6)
    next_state = model.forward_conditional.sample(pf_params)
    logpf[:, 1] = model.forward_conditional.log_prob(next_state, pf_params)

    logpf.sum().backward()

    print(init_states.grad)


def test_get_forward_trajectory(model_name: str):
    energy, model, cfg = load_energy_model_and_config(
        model_name=model_name,
    )

    init_states = model.generate_initial_state(300)
    traj, logpf, logpb = model.get_forward_trajectory(init_states)

    print(logpf, logpb)


if __name__ == "__main__":
    test_get_forward_trajectory("DoubleGFN")
