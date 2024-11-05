import abc
import torch

from typing import Optional, Callable


"""
------- Guide for new energy function implementation -------

Make new energy function as subclass as BaseEnergy.

Then implement the followings:
    1. Set the value of class variable logZ_is_available, can_sample.

    2. _ground_truth_logZ must be set if logZ_is_available.

    3. Implement energy method.
    
    4. Implement _generate_sample method if can_sample.

    5. device and dimension must be set (by BaseEnergy constructor).
"""


class BaseEnergy(abc.ABC):

    logZ_is_available: bool = False
    can_sample: bool = False

    def __init__(self, device, dim):
        self.device = device
        self.data_ndim = dim

    @property
    def ground_truth_logZ(self):
        if not self.logZ_is_available:
            raise Exception("log Z is not available for this energy function")
        return self._ground_truth_logZ

    @staticmethod
    def _match_device(func: Callable[[torch.Tensor], torch.Tensor]):
        def wrapper(self, x: torch.Tensor):
            if x.device != self.device:
                device = x.device
                x = x.to(device=self.device)
            return func(self, x).to(device=device)

        return wrapper

    @abc.abstractmethod
    def energy(self, x: torch.Tensor):
        return

    def unnormalized_density(self, x: torch.Tensor):
        return torch.exp(-self.energy(x))

    @property
    def ndim(self):
        return self.data_ndim

    @abc.abstractmethod
    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        pass

    def sample(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """
        Generate ground truth sample from energy function.

        Args:
            batch_size (int): Number of sample to generate.

        Returns:
            torch.Tensor: generated sample.
        """

        if not self.can_sample:
            raise Exception(
                "Ground truth sample is not available for this energy function"
            )

        if device is None:
            device = self.device

        return self._generate_sample(batch_size).to(device=device)

    @_match_device
    def score(self, x: torch.Tensor):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                (-self.energy(copy_x)).sum().backward()
                grad_energy = copy_x.grad.data
            return grad_energy

    def log_reward(self, x: torch.Tensor):
        return -self.energy(x)

    def cached_sample(self, batch_size: int, do_resample: bool = False):
        sample_is_needed = not hasattr(
            self, "_cached_sample"
        ) or batch_size != self._cached_sample.size(0)

        if sample_is_needed or do_resample:
            self._cached_sample = self.sample(batch_size)
        return self._cached_sample
