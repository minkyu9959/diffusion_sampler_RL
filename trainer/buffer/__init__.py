from .prioritized_buffer import (
    RankPrioritizedReplayBuffer,
    RewardPrioritizedReplayBuffer,
)
from .state_transition_buffer import StateTransitionReplayBuffer
from .simple_buffer import SimpleReplayBuffer
from .base_buffer import BaseBuffer

from typing import Optional
from omegaconf import DictConfig

from energy import BaseEnergy


def get_buffer(
    buffer_cfg: Optional[DictConfig], energy_function: BaseEnergy
) -> Optional[BaseBuffer]:

    if buffer_cfg is None:
        return None

    if buffer_cfg.prioritized == "rank":
        buffer_class = RankPrioritizedReplayBuffer
    elif buffer_cfg.prioritized == "reward":
        buffer_class = RewardPrioritizedReplayBuffer
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
