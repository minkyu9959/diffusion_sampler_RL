class Logger:
    def log_loss(self, loss: dict):
        raise NotImplementedError

    def log_visual(self, visuals: dict, epoch: int):
        raise NotImplementedError

    def log_metric(self, metrics: dict, epoch: int):
        raise NotImplementedError
