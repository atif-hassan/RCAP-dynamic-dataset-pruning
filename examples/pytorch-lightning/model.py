from torchvision.models import resnet18
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch





class MyModel(pl.LightningModule):
    def __init__(self, num_classes=10, logger_object=None):
        super(MyModel, self).__init__()
        self.backbone = resnet18()
        self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features, out_features=num_classes)
        self.logger_object = logger_object
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.train_acc, self.val_acc = Accuracy(task="multiclass", num_classes=num_classes), Accuracy(task="multiclass", num_classes=num_classes, average="none")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=5e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
            }
        }


    def forward(self, x):
        out = self.backbone(x)
        return out


    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        # If the logger object is provided, log the metrics
        if self.logger_object is not None:
            self.logger_object.log_metric(epoch=self.current_epoch, sample_idx=idx, metric=preds)
        self.train_acc.update(preds, y)
        self.log("train", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("loss", torch.mean(loss), on_step=True, prog_bar=True)
        return torch.mean(loss)


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.forward(x)
        self.val_acc.update(preds, y)