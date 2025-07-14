import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from typing import Dict, Any
import torchmetrics
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class LSTMClassifier(pl.LightningModule):
    """
    Classificador LSTM para vídeos.
    Usa um CNN pre-treinado para extrair features dos frames
    e um LSTM para classificar a sequência.
    """
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cnn_backbone: str = 'resnet18',
        freeze_cnn: bool = True,
        class_names: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_names = class_names
        
        # CNN backbone para extração de features
        if cnn_backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Identity()  # Remover classificador
            cnn_output_size = 512
        elif cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            self.cnn.fc = nn.Identity()
            cnn_output_size = 2048
        else:
            raise ValueError(f"Backbone CNN não suportado: {cnn_backbone}")
        
        # Congelar CNN se solicitado
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Projeção das features do CNN para o tamanho de entrada do LSTM
        self.feature_projection = nn.Linear(cnn_output_size, input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Classificador final
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
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
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai features dos frames usando CNN
        Args:
            x: tensor de shape (batch_size, seq_len, channels, height, width)
        Returns:
            features: tensor de shape (batch_size, seq_len, feature_size)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape para processar todos os frames juntos
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extrair features com CNN
        with torch.set_grad_enabled(not self.hparams.freeze_cnn):
            features = self.cnn(x)
        
        # Projetar features para o tamanho desejado
        features = self.feature_projection(features)
        
        # Reshape de volta para sequência
        features = features.view(batch_size, seq_len, -1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: tensor de shape (batch_size, seq_len, channels, height, width)
        Returns:
            logits: tensor de shape (batch_size, num_classes)
        """
        # Extrair features dos frames
        features = self.extract_features(x)
        
        # Processar sequência com LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Usar o último estado oculto para classificação
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        # Pegamos a última saída temporal
        final_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Classificação
        logits = self.classifier(final_output)
        
        return logits
    
    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = getattr(self, f"{stage}_acc")
        f1 = getattr(self, f"{stage}_f1")
        cm = getattr(self, f"{stage}_cm")

        acc(preds, y)
        f1(preds, y)
        cm.update(preds, y)

        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_f1', f1, on_step=False, on_epoch=True)
        
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        return self._step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step"""
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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
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
    
    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Prediction step"""
        x, _ = batch
        logits = self(x)
        return torch.softmax(logits, dim=1)
