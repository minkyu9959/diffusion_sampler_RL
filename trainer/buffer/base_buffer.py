import abc

from typing import Callable

import torch


class BaseBuffer(abc.ABC):
    def __init__(
        self,
        buffer_size: int,
        device: str,
        log_reward_function: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        data_ndim: int,
        beta=1.0,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, x: torch.Tensor):
        return

    @abc.abstractmethod
    def sample(self):
        return
