# models/frame_classifier.py
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
import torchmetrics
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class FrameClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        cnn_backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        scheduler_name: str = None,
        t_max: int = 10,
        eta_min: float = 1e-5,
        class_names: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone (pode ser resnet, vit, etc.)
        self.backbone = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            num_classes=0  # sem FC
        )

        # Congelar se necessário
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.t_max = t_max
        self.eta_min = eta_min
        self.class_names = class_names

        # Métricas
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

        self.train_cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.val_cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.test_cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Entrada: B x T x C x H x W (frames)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.backbone(x)
            x = feats.view(B, T, -1).mean(dim=1)  # Média dos frames
        else:
            x = self.backbone(x)
        return self.classifier(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.softmax(dim=-1)

        acc = getattr(self, f"{stage}_acc")(preds, y)
        f1 = getattr(self, f"{stage}_f1")(preds, y)
        cm = getattr(self, f"{stage}_cm")
        cm.update(preds, y)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _epoch_end(self, stage):
        cm = getattr(self, f"{stage}_cm")
        cm_tensor = cm.compute()
        cm.reset()

        if self.trainer.logger and hasattr(self.trainer.logger.experiment, 'log'):
            fig = self._plot_confusion_matrix(cm_tensor.cpu().numpy(), self.class_names)
            self.trainer.logger.experiment.log({
                f'{stage}_confusion_matrix': wandb.Image(fig)
            })
            plt.close(fig)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def _plot_confusion_matrix(self, cm, class_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        return fig

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_.max, eta_min=self.eta_min)
            return [optimizer], [scheduler]
        return optimizer
