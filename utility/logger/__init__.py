from .base_logger import Logger
from .neptune_logger import NeptuneLogger
from .wandb_logger import WandbLogger

from dotenv import load_dotenv, find_dotenv

# Load .env file to get environment variables
# This is necessary to use environment variables in the logger code
load_dotenv(find_dotenv(), override=True)


class NullLogger(Logger):
    # do nothing logger
    def __init__(self):
        self.detail_log = False

    def log_loss(self, loss: dict, epoch: int):
        pass

    def log_visual(self, visuals: dict, epoch: int):
        pass

    def log_metric(self, metrics: dict, epoch: int):
        pass

    def log_model(self, model, epoch: int, is_final: bool = False):
        pass

    def log_gradient(self, model, epoch: int):
        pass

    def finish(self):
        pass


def get_logger(cfg, output_dir, **kwargs):
    if cfg.logger == "neptune":
        return NeptuneLogger(cfg=cfg, output_dir=output_dir, **kwargs)
    elif cfg.logger == "wandb":
        return WandbLogger(cfg=cfg, output_dir=output_dir, **kwargs)
    elif cfg.logger == None:
        return NullLogger()
    else:
        raise Exception("Invalid logger")
