import torch
from torch.utils.data.sampler import Sampler

from torch.distributions.gumbel import Gumbel
import numpy as np

from typing import (
    Iterator,
    Optional,
    Sequence,
    Sized,
)

from .simple_buffer import SimpleReplayBuffer

"""
Here, we implement replay buffer with torch data loader + weighted random sampler.
"""


class RankPrioritizedReplayBuffer(SimpleReplayBuffer):
    def make_sampler(self):
        # With replacement, elements are sampled with probability
        # inversely proportional to rank of energy.

        rewards_np = self.storage.rewards.detach().cpu().view(-1).numpy()
        ranks = np.argsort(np.argsort(-1 * rewards_np))
        weights = 1.0 / (1e-2 * len(rewards_np) + ranks)

        return torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(self.storage), replacement=True
        )


class LogitSampler(Sampler):
    data_source: Sized
    replacement: bool = True

    def __init__(
        self,
        logits: Sequence[float],
        num_samples: Optional[int] = None,
    ):
        self.logits = logits
        self._num_samples = num_samples
        self.gumbel = torch.distributions.gumbel.Gumbel(
            torch.tensor([0.0]), torch.tensor([1.0])
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        while True:
            gumbel_sample = (
                self.gumbel.sample((len(self.logits),)).view(-1).to(self.logits)
            )
            yield torch.argmax(self.logits + gumbel_sample)

    def __len__(self) -> int:
        return self.num_samples


class RewardPrioritizedReplayBuffer(SimpleReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel_dist = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))

    def make_sampler(self):
        # With replacement, elements are sampled with probability proportional to energy.

        log_rewards = self.storage.rewards.detach().view(-1)
        return LogitSampler(log_rewards, num_samples=len(self.storage))
