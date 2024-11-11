import torch

from utility.loader import load_all_from_experiment_path


PATH = "results/outputs/2024-10-08/10-16-33"


def main() -> None:
    """
    Main entry point for finetuning.
    """
    energy, model, trainer, _ = load_all_from_experiment_path(
        PATH, must_init_logger=True
    )

    trainer.train_cfg.fwd_loss = "tb"
    trainer.max_epoch = trainer.train_cfg.epochs = 10000

    for g in trainer.optimizer.param_groups:
        g["lr"] = 1e-4

    model.logZ = torch.nn.Parameter(
        model.logZ_ratio.detach().sum(dim=0) + model.prior_energy.ground_truth_logZ
    )

    trainer.optimizer.add_param_group(
        {
            "params": model.logZ,
            "lr": 1e-1,
        }
    )

    trainer.train()


if __name__ == "__main__":
    main()
