import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from typing import Dict, Any
import torchmetrics
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class ViTClassifier(pl.LightningModule):
    """
    Classificador Vision Transformer para vídeos.
    Processa cada frame individualmente e depois agrega os resultados.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 7,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        aggregation_method: str = 'mean',
        dropout: float = 0.1,
        class_names: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.aggregation_method = aggregation_method
        self.class_names = class_names
        
        # Carregar modelo ViT pre-treinado
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Sem classificador final
            global_pool=''  # Sem pooling global
        )
        
        # Congelar backbone se solicitado
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Obter dimensão das features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.vit(dummy_input)
            if len(features.shape) == 3:  # (batch, seq_len, feature_dim)
                feature_dim = features.shape[-1]
            else:  # (batch, feature_dim)
                feature_dim = features.shape[-1]
        
        # Agregação temporal
        if aggregation_method == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        elif aggregation_method == 'lstm':
            self.temporal_lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=feature_dim // 2,
                num_layers=2,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
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
    
    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai features de cada frame usando ViT
        Args:
            x: tensor de shape (batch_size, seq_len, channels, height, width)
        Returns:
            features: tensor de shape (batch_size, seq_len, feature_dim)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape para processar todos os frames juntos
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extrair features com ViT
        features = self.vit(x)
        
        # Se o ViT retorna sequência de patches, fazer pooling
        if len(features.shape) == 4:
    # flatten de B×C×H×W para B×(C·H·W)
            features = features.view(features.size(0), -1)

        
        # Reshape de volta para sequência
        features = features.view(batch_size, seq_len, -1)
        
        return features
    
    def aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Agrega as features dos frames ao longo do tempo
        Args:
            features: tensor de shape (batch_size, seq_len, feature_dim)
        Returns:
            aggregated_features: tensor de shape (batch_size, feature_dim)
        """
        if self.aggregation_method == 'mean':
            return torch.mean(features, dim=1)
        elif self.aggregation_method == 'max':
            return torch.max(features, dim=1)[0]
        elif self.aggregation_method == 'attention':
            # Usar o [CLS] token como query para a atenção
            cls_token = features[:, 0, :].unsqueeze(1)
            attn_output, _ = self.temporal_attention(cls_token, features, features)
            return attn_output.squeeze(1)
        elif self.aggregation_method == 'lstm':
            lstm_out, (h_n, c_n) = self.temporal_lstm(features)
            # Usar a última saída temporal
            return lstm_out[:, -1, :]
        else:
            raise ValueError(f"Método de agregação não suportado: {self.aggregation_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: tensor de shape (batch_size, seq_len, channels, height, width)
        Returns:
            logits: tensor de shape (batch_size, num_classes)
        """
        # Extrair features dos frames
        frame_features = self.extract_frame_features(x)
        
        # Agregar features ao longo do tempo
        aggregated_features = self.aggregate_features(frame_features)
        
        # Classificação
        logits = self.classifier(aggregated_features)
        
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
    
    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Prediction step"""
        x, _ = batch
        logits = self(x)
        return torch.softmax(logits, dim=1)
