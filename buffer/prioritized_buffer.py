from .simple_buffer import SimpleReplayBuffer

import torch
import numpy as np

"""
Here, we implement replay buffer with torch data loader + weighted random sampler.
"""


class PrioritizedReplayBuffer(SimpleReplayBuffer):
    def make_sampler(self):
        # With replacement, elements are sampled with probability proportional to energy.

        self.scores_numpy = self.storage.rewards.detach().cpu().view(-1).numpy()
        ranks = np.argsort(np.argsort(-1 * self.scores_numpy))
        weights = 1.0 / (1e-2 * len(self.scores_numpy) + ranks)

        return torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(self.storage), replacement=True
        )
