import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from typing import Dict, Any
import torchmetrics

class Attention(nn.Module):
    """Simple Self-Attention mechanism."""
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)

        # Weighted sum
        context = torch.bmm(attention_weights, v)
        return context, attention_weights

class ImprovedLSTMClassifier(pl.LightningModule):
    """
    Improved GRU-based classifier with Attention for videos.
    """
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.4,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        cnn_backbone: str = 'resnet18',
        freeze_cnn: bool = True,
        scheduler_config: Dict[str, Any] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config
        
        # CNN backbone
        if cnn_backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Identity()
            cnn_output_size = 512
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            # Fine-tuning: unfreeze later layers
            for name, param in self.cnn.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        self.feature_projection = nn.Linear(cnn_output_size, input_size)
        
        # GRU layers
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        rnn_output_size = hidden_size * 2  # Bidirectional
        
        # Attention layer
        self.attention = Attention(rnn_output_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        with torch.set_grad_enabled(not self.hparams.freeze_cnn):
            features = self.cnn(x)
            
        features = self.feature_projection(features)
        features = features.view(batch_size, seq_len, -1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        
        rnn_out, _ = self.rnn(features)
        
        # Apply attention
        attn_out, _ = self.attention(rnn_out)
        
        # Global average pooling over the sequence dimension
        pooled_out = attn_out.mean(dim=1)
        
        logits = self.classifier(pooled_out)
        return logits
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_config is None or self.scheduler_config.name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.2,
                patience=5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        elif self.scheduler_config.name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.t_max,
                eta_min=self.scheduler_config.eta_min
            )
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_config.name}")
