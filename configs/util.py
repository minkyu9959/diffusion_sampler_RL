from omegaconf import DictConfig


def get_trainer_name_from_config(cfg: DictConfig) -> str:
    return cfg.train.trainer._target_.split(".")[-1].replace("Trainer", "")


def get_model_name_from_config(cfg: DictConfig) -> str:
    return cfg.model._target_.split(".")[-1]


def get_energy_name_from_config(cfg: DictConfig) -> str:
    return cfg.energy.name


def make_tag(cfg: DictConfig):

    tags = []

    if cfg.train.get("local_search"):
        tags.append("LS")

    if cfg.train.get("exploratory"):
        tags.append("Expl")

    if cfg.model.get("langevin_scaler"):
        tags.append("LP")

    if cfg.train.get("fwd_loss"):
        tags.append(f"fwd_{cfg.train.fwd_loss}")

    if cfg.train.get("bwd_loss"):
        tags.append(f"bwd_{cfg.train.bwd_loss}")

    return tags
