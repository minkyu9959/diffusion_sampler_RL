import math


def log_mean_exp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def add_prefix_to_dict_key(prefix: str, dict: dict) -> dict:
    return {f"{prefix}/name": dict[name] for name in dict}
