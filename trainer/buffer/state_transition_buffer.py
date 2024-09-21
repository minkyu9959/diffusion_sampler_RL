import torch

from .storage import TransitionStorage


class StateTransitionReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        device: str,
        batch_size: int,
        data_ndim: int,
        num_time_steps: int,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.batch_size = batch_size
        self.data_ndim = data_ndim
        self.num_time_steps = num_time_steps

    def add(self, states: torch.Tensor, time: torch.Tensor):
        states = states.detach()

        if not hasattr(self, "storage"):
            self.storage = TransitionStorage(states, time)
        else:
            self.storage.update(states, time)

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
        weights = torch.ones(len(self.storage))

        return torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(self.storage), replacement=True
        )

    def sample(self):
        try:
            states, time = next(self.data_iter)
        except:
            self.data_iter = iter(self.loader)
            states, time = next(self.data_iter)

        return states.detach(), time.detach()
