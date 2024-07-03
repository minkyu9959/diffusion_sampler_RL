import torch


class SampleRewardStorage(torch.utils.data.Dataset):
    def __init__(self, sample: torch.Tensor, rewards: torch.Tensor):
        super().__init__()

        if not self.__has_same_batch_size(sample, rewards):
            raise Exception("Batch size not matches")

        self.sample = sample
        self.rewards = rewards

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample[idx], self.rewards[idx]

    def update(self, sample: torch.Tensor, rewards: torch.Tensor):
        if not self.__has_same_batch_size(sample, rewards):
            raise Exception("Batch size not matches")

        self.sample = torch.cat([self.sample, sample], dim=0)
        self.rewards = torch.cat([self.rewards, rewards], dim=0)

    def deque(self, num_elem_to_delete: int):
        self.sample = self.sample[num_elem_to_delete:]
        self.rewards = self.rewards[num_elem_to_delete:]

    def __has_same_batch_size(self, sample: torch.Tensor, rewards: torch.Tensor):
        return sample.size(0) == rewards.size(0)

    def __len__(self):
        assert self.rewards.size(0) == self.sample.size(0)

        return self.sample.size(0)

    def collate(batched_sample_and_rewards: list[tuple[torch.Tensor, torch.Tensor]]):
        raise NotImplementedError
        # return torch.stack(data_list)
