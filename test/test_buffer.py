import torch

from buffer import *
from energy import *

energy_function = GMM25("cuda:0")


def test_simple_replay_buffer():
    buffer_len = 100
    batch_size = 10

    buffer = SimpleReplayBuffer(
        buffer_len, "cpu", energy_function.log_reward, batch_size, 2, 1.0
    )

    buffer.add(torch.randn(30, 2), torch.randn(30))

    sample, reward = buffer.sample()
    assert sample.size(0) == batch_size and reward.size(0) == batch_size

    buffer.add(torch.zeros(60, 2), torch.zeros(60))
    buffer.add(torch.randn(30, 2), torch.randn(30))

    sample, reward = buffer.sample()

    assert sample.size(0) == batch_size and reward.size(0) == batch_size

    assert 0 in sample


def test_prioritized_replay_buffer():
    buffer_len = 100
    batch_size = 10

    buffer = PrioritizedReplayBuffer(
        buffer_len, "cpu", energy_function.log_reward, batch_size, 2, 1.0
    )

    buffer.add(torch.ones(5, 2), torch.ones(5))
    buffer.add(torch.zeros(60, 2), torch.zeros(60))
    buffer.add(torch.randn(30, 2), torch.abs(torch.randn(30)))

    sample, reward = buffer.sample()

    # check if buffer sample has proper shape.
    assert sample.size(0) == batch_size and reward.size(0) == batch_size

    # check if buffer sample follows the priority.
    assert (reward > 0.5).sum() >= 3
