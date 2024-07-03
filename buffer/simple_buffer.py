from typing import Callable

from .storage import SampleRewardStorage

import torch


"""
Here, we implement replay buffer with torch data loader + weighted random sampler.
"""


class SimpleReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        device: str,
        log_reward_function: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        data_ndim: int,
        beta=1.0,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.log_reward_function = log_reward_function
        self.batch_size = batch_size
        self.data_ndim = data_ndim
        self.beta = beta

    def add(self, sample: torch.Tensor, log_reward: torch.Tensor):
        sample = sample.detach()
        log_reward = log_reward.detach()

        if not hasattr(self, "storage"):
            self.storage = SampleRewardStorage(sample, log_reward)
        else:
            self.storage.update(sample, log_reward)

        # If buffer is full, delete some elements.
        num_elem_in_storage = len(self.storage)
        if num_elem_in_storage > self.buffer_size:
            num_elem_to_delete = num_elem_in_storage - self.buffer_size
            self.storage.deque(num_elem_to_delete)

        # Sampler provides priority information to data loader.
        self.sampler = self.make_sampler()

        self.loader = torch.utils.data.DataLoader(
            self.storage,
            sampler=self.sampler,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def make_sampler(self):
        # With replacement, all elements are sampled with same probability.
        weights = torch.ones(len(self.storage))

        return torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(self.storage), replacement=True
        )

    def sample(self):
        try:
            sample, reward = next(self.data_iter)
        except:
            self.data_iter = iter(self.loader)
            sample, reward = next(self.data_iter)

        return sample.detach(), reward.detach()
