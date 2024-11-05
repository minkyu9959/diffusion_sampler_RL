class Logger:
    def log_loss(self, loss: dict, epoch: int):
        raise NotImplementedError

    def log_visual(self, visuals: dict, epoch: int):
        raise NotImplementedError

    def log_metric(self, metrics: dict, epoch: int):
        raise NotImplementedError

    def log_model(self, model, epoch: int, is_final: bool = False):
        raise NotImplementedError
