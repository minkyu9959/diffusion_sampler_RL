from hydra import compose, initialize

import torch

from train import set_seed
from models import get_model
from energy import get_energy_function

from models import GFN


def test_stochastic_backprop():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            overrides=["model=GFN", "model.forward_model.zero_init=false"],
        )

    set_seed(cfg.seed)

    energy = get_energy_function(cfg)

    gfn: GFN = get_model(cfg, energy).to(cfg.device)

    state = init_states = torch.ones(
        (1, energy.ndim), device=cfg.device, requires_grad=True
    )
    logpf = torch.zeros((1, 2), device=cfg.device)

    # First step
    next_state, pf_params = gfn.get_next_state(state, 0.5, stochastic_backprop=True)

    logpf[:, 0] = gfn.get_forward_logprob(next_state, state, pf_params)

    # Second step
    state = next_state
    next_state, pf_params = gfn.get_next_state(state, 0.5, stochastic_backprop=True)
    logpf[:, 1] = gfn.get_forward_logprob(next_state, state, pf_params)

    logpf.sum().backward()

    print(init_states.grad)


if __name__ == "__main__":
    test_stochastic_backprop()
