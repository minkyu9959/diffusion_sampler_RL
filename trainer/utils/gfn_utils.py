from typing import Optional

import torch

import numpy as np

from buffer import *
from energy import BaseEnergy

from omegaconf import DictConfig


def get_buffer(
    buffer_cfg: Optional[DictConfig], energy_function: BaseEnergy
) -> Optional[BaseBuffer]:

    if buffer_cfg is None:
        return None

    if buffer_cfg.prioritized:
        buffer_class = PrioritizedReplayBuffer
    else:
        buffer_class = SimpleReplayBuffer

    return buffer_class(
        buffer_size=buffer_cfg.buffer_size,
        device=buffer_cfg.device,
        log_reward_function=energy_function.log_reward,
        batch_size=buffer_cfg.batch_size,
        data_ndim=energy_function.data_ndim,
        beta=buffer_cfg.beta,
    )


def get_exploration_std(
    epoch, exploratory, exploration_factor=0.1, exploration_wd=False
):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1.0 - epoch / 5000.0)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl
