from hydra import compose, initialize

import torch

from train import set_seed
from models import get_model
from energy import get_energy_function, Plotter

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


def test_get_forward_trajectory():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            overrides=["model=GFN", "model.forward_model.zero_init=false"],
        )

    set_seed(cfg.seed)

    energy = get_energy_function(cfg)

    gfn: GFN = get_model(cfg, energy).to(cfg.device)

    init_states = gfn.generate_initial_state(300)
    traj, logpf, logpb = gfn.get_forward_trajectory(init_states)

    print(logpf, logpb)


def load_model_and_debug(model_path: str):
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            overrides=[
                "experiment=AnnealedGFN/AnnealedDB+LP",
                "energy=25gmm",
            ],
        )

    set_seed(cfg.seed)

    energy = get_energy_function(cfg)

    plotter = Plotter(energy, **cfg.eval.plot)

    gfn: GFN = get_model(cfg, energy).to(cfg.device)

    gfn.load_state_dict(torch.load(model_path))

    init_states = gfn.generate_initial_state(300)
    traj, logpf, logpb = gfn.get_forward_trajectory(init_states)

    anim, _, _ = plotter.make_sample_generation_animation(traj)
    anim.save("test/traj_animation.gif")


if __name__ == "__main__":
    load_model_and_debug("test/models/model.pt")
