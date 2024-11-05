import wandb

import math
from typing import Union

import matplotlib.pyplot as plt
import torch

from .base_logger import Logger

from configs.util import *


class WandbLogger(Logger):
    def __init__(
        self, cfg: Union[dict, DictConfig], output_dir: str, debug: bool = False
    ):

        # require more detailed log for debugging purpose
        self.detail_log = debug

        # neptune logger cannot accept OmegaConf object.
        # Convert it to python dictionary.
        if type(cfg) is DictConfig:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg_dict = cfg

        group = cfg_dict.pop("group", None)
        name = cfg_dict.pop("name", None)

        self.run = wandb.init(
            project="Diffusion-sampler",
            entity="dywoo1247",
            tags=make_tag(cfg),
            name=name,
            group=group,
            config={
                **cfg_dict,
                "output_dir": output_dir,
            },
        )

        self.output_dir = output_dir

    def log_loss(self, loss: dict, epoch: int):
        loss = {f"train/{k}": v for k, v in loss.items()}

        self.run.log(loss, step=epoch)

    def log_metric(self, metrics: dict, epoch: int):

        metrics = {
            f"eval/{k}": (v if not math.isinf(v) and not math.isnan(v) else 0.0)
            for k, v in metrics.items()
        }

        self.run.log(metrics, step=epoch)

    def log_visual(self, visuals: dict, epoch: int):
        for visual_name, fig in visuals.items():
            fig.savefig(f"{self.output_dir}/{visual_name}.pdf", bbox_inches="tight")

            self.run.log({f"visuals/{visual_name}": wandb.Image(fig)}, step=epoch)

        # Prevent too much plt objects from lasting
        plt.close("all")

    def log_model(self, model: torch.nn.Module, epoch: int, is_final: bool = False):
        final = "_final" if is_final else ""

        model_path = f"{self.output_dir}/model{final}.pt"

        torch.save(model.state_dict(), model_path)
        self.run.log_model(path=model_path)

    def log_gradient(self, model: torch.nn.Module, epoch: int):
        assert self.detail_log

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.run.log(
                    {
                        f"gradient/{name}_max": param.grad.max().detach().cpu().numpy(),
                    },
                    step=epoch,
                )

    def finish(self):
        if hasattr(self, "run"):
            wandb.finish()
