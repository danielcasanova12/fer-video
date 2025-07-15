
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from typing import Dict, Any

class YOLOClassifier(pl.LightningModule):
    """
    Classificador baseado em YOLOv8 para vídeos.
    Processa cada frame individualmente e agrega as probabilidades.
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8l-cls.pt', # Default YOLOv8 Large Classification
        num_classes: int = 7,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        class_names: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_names = class_names

        # Carregar modelo YOLOv8 de classificação
        self.yolo_model = YOLO(model_name)

        # Congelar o backbone do YOLO se solicitado
        if freeze_backbone:
            for param in self.yolo_model.parameters():
                param.requires_grad = False

        # O YOLOv8-cls já produz logits/probabilidades para 1000 classes (ImageNet).
        # Precisamos de uma camada linear para adaptar ao nosso num_classes.
        # A camada final do YOLOv8-cls é `model.model[-1]`. Se for `C2f_cls`, ela tem um `cv3` que é a camada linear final.
        # Vamos inspecionar o modelo para pegar a dimensão de entrada correta.
        # Uma forma segura é rodar um forward pass dummy e pegar a saída.
        dummy_input = torch.randn(1, 3, 224, 224) # YOLOv8 espera 224x224 por padrão para cls
        with torch.no_grad():
            # O método predict retorna um objeto Results, que contém as probabilidades
            results = self.yolo_model.predict(dummy_input, verbose=False)
            # As probabilidades estão em results[0].probs.data
            yolo_output_dim = results[0].probs.data.shape[0] # Deve ser 1000 para ImageNet

        # Se o número de classes do YOLO for diferente do nosso, adicionamos uma camada linear
        if yolo_output_dim != num_classes:
            self.final_classifier = nn.Sequential(
                nn.Linear(yolo_output_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        else:
            # Se as classes coincidirem, usamos uma camada de identidade (ou nenhuma, se o YOLO já for o classificador final)
            self.final_classifier = nn.Identity()

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
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # Processar todos os frames juntos

        # YOLOv8 predict retorna uma lista de objetos Results
        # Cada Results object tem um atributo .probs para classificação
        yolo_results = self.yolo_model.predict(x, verbose=False) # verbose=False para não poluir o log
        
        # Extrair as probabilidades (logits) de cada resultado e empilhar
        # results[0].probs.data é um tensor de probabilidades/logits
        frame_outputs = torch.stack([r.probs.data for r in yolo_results])

        # Agregação temporal: Média das probabilidades/logits dos frames
        # Reshape de volta para (B, T, num_classes) e então média sobre T
        aggregated_output = frame_outputs.view(B, T, -1).mean(dim=1)

        # Passar pela camada classificadora final (se houver)
        logits = self.final_classifier(aggregated_output)
        
        return logits

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

        # YOLOClassifier pode ter um scheduler diferente ou nenhum
        # Por simplicidade, vamos usar um ReduceLROnPlateau padrão se não for especificado
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
