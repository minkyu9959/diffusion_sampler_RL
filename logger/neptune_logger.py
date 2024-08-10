import neptune

import sys

import matplotlib.pyplot as plt
import torch

from .base_logger import Logger

from configs.util import *


class NeptuneLogger(Logger):
    def __init__(self, cfg: dict, output_dir: str):
        self.run = neptune.init_run(
            project="dywoo1247/Diffusion-sampler",
            tags=make_tag(cfg),
            dependencies="infer",
            name=cfg.get("name"),
        )

        if group_tag := cfg.get("group_tag"):
            self.run["sys/group_tags"].add(group_tag)

        # neptune logger cannot accept OmegaConf object.
        # Convert it to python dictionary.
        self.run["parameters"] = OmegaConf.to_container(cfg)
        self.run["scripts"] = "python3 " + " ".join(sys.argv)

        self.run["trainer"] = get_trainer_name_from_config(cfg)
        self.run["model"] = get_model_name_from_config(cfg)
        self.run["energy"] = get_energy_name_from_config(cfg)

        self.output_dir = output_dir

    def log_loss(self, loss: float):
        self.run["train/loss"].append(loss)

    def log_metric(self, metrics: dict, epoch: int):
        for metric_name, metric_value in metrics.items():
            self.run[f"eval/{metric_name}"].append(value=metric_value, step=epoch)

    def log_visual(self, visuals: dict, epoch: int):
        for visual_name, fig in visuals.items():
            fig.savefig(f"{self.output_dir}/{visual_name}.pdf", bbox_inches="tight")

            self.run[f"visuals/{visual_name}"].append(fig, step=epoch)

        # Prevent too much plt objects from lasting
        plt.close("all")

    def log_model(self, model: torch.nn.Module, epoch: int, is_final: bool = False):
        final = "_final" if is_final else ""

        model_path = f"{self.output_dir}/model{final}.pt"

        torch.save(model.state_dict(), model_path)
        self.run[f"model_ckpts/epoch_{epoch}"].upload(model_path)

    def __del__(self):
        if hasattr(self, "run"):
            self.run.stop()
