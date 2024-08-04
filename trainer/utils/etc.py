import torch

import PIL

import hydra
from omegaconf import DictConfig, OmegaConf


OUTPUT_DIR = None


def set_experiment_output_dir():
    global OUTPUT_DIR
    OUTPUT_DIR = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


def get_experiment_output_dir():
    return OUTPUT_DIR


def make_tag(cfg: DictConfig):
    trainer_name = cfg.train.trainer._target_.split(".")[-1].replace("Trainer", "")
    model_name = cfg.model._target_.split(".")[-1]

    tags = [trainer_name, model_name]

    if cfg.train.get("local_search"):
        tags.append("LS")

    if cfg.train.get("exploratory"):
        tags.append("Expl")

    if cfg.train.get("fwd_loss"):
        tags.append(f"fwd_{cfg.train.fwd_loss}")

    if cfg.train.get("bwd_loss"):
        tags.append(f"bwd_{cfg.train.bwd_loss}")

    return tags


def check_config_and_set_read_only(cfg: DictConfig):

    # From now, config file cannot be modified.
    OmegaConf.set_readonly(cfg, True)
