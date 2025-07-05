import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from typing import Dict, Any
import torchmetrics


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
        dropout: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.aggregation_method = aggregation_method
        
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