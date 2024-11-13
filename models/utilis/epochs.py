import segmentation_models_pytorch.utils as smp_utils
import torch
class TrainEpoch(smp_utils.train.TrainEpoch):
    def on_epoch_start(self):
        self.model.train()

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, scheduler=None, amp=False, get_prediction=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
        )
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.amp = torch.amp.grad_scaler.GradScaler() if amp else None
        self.get_prediction = get_prediction

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=self.amp is not None):
            y_pred = self.get_prediction(self.model(x))
            loss = self.loss(y_pred, y)
        if self.amp is not None:
            self.amp.scale(loss).backward()
            self.amp.step(self.optimizer)
            self.amp.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        return loss, y_pred

class ValidEpoch(smp_utils.train.ValidEpoch):
    def on_epoch_start(self):
        self.model.eval()

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, get_prediction=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=verbose,
        )
        self.get_prediction = get_prediction

    def batch_update(self, x, y):
        with torch.no_grad():
            y_pred = self.get_prediction(self.model(x))
            loss = self.loss(y_pred, y)
        return loss, y_pred
    
class TestEpoch(smp_utils.train.ValidEpoch):
    def on_epoch_start(self):
        self.model.eval()

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, get_prediction=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=verbose,
        )
        self.get_prediction = get_prediction

    def batch_update(self, x, y):
        with torch.no_grad():
            y_pred = self.get_prediction(self.model(x))
            loss = self.loss(y_pred, y)
        return loss, y_pred