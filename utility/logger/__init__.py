from .base_logger import Logger
from .neptune_logger import NeptuneLogger
from .wandb_logger import WandbLogger


def get_logger(cfg, output_dir, **kwargs):
    if cfg.logger == "neptune":
        return NeptuneLogger(cfg=cfg, output_dir=output_dir, **kwargs)
    elif cfg.logger == "wandb":
        return WandbLogger(cfg=cfg, output_dir=output_dir, **kwargs)
