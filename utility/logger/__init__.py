from .base_logger import Logger
from .neptune_logger import NeptuneLogger


def get_logger(cfg, output_dir, **kwargs):
    return NeptuneLogger(cfg=cfg, output_dir=output_dir, **kwargs)
