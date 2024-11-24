import torch
from typing import Union

from ..models import GFN
from ..models import CMCDSampler

    
def jarzynski_reverse_KL(
    sampler: Union[GFN, CMCDSampler],
):
    if isinstance(sampler, GFN):
        learnable_drift = sampler.forward_policy
    elif isinstance(sampler, CMCDSampler):
        learnable_drift = sampler.control_model
    
    infinitesimal_work = (learnable_drift.norm_traj - sampler.partial_t_log_prob_traj) * sampler.dt
    loss = (infinitesimal_work.sum(-1)).mean()
    
    return loss