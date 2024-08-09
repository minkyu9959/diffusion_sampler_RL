from .simple_buffer import SimpleReplayBuffer

import torch
import numpy as np

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


class RewardPrioritizedReplayBuffer(SimpleReplayBuffer):
    def make_sampler(self):
        # With replacement, elements are sampled with probability proportional to energy.

        weights = np.exp(self.storage.rewards.detach().cpu().view(-1).numpy())

        return torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(self.storage), replacement=True
        )
