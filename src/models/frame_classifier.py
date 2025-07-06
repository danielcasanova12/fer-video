import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class FrameClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float, weight_decay: float,
                 cnn_backbone: str = 'resnet50', pretrained: bool = True,
                 freeze_backbone: bool = True, scheduler_name: str = None,
                 t_max: int = None, eta_min: float = None):
        super().__init__()
        self.save_hyperparameters()

        # Carregar backbone CNN
        if cnn_backbone == 'resnet18':
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity() # Remover a camada final de classificação
        elif cnn_backbone == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone CNN não suportado: {cnn_backbone}")

        # Congelar backbone se especificado
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Cabeça de classificação
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.t_max = t_max
        self.eta_min = eta_min

    def forward(self, x):
        # x é (batch_size, num_frames, C, H, W)
        batch_size, num_frames, C, H, W = x.size()

        # Redimensionar para (batch_size * num_frames, C, H, W) para passar pela CNN
        x = x.view(batch_size * num_frames, C, H, W)

        # Extrair features de cada frame
        features = self.feature_extractor(x) # (batch_size * num_frames, feature_dim)

        # Redimensionar de volta para (batch_size, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # Agregação por média (average pooling)
        # (batch_size, feature_dim)
        aggregated_features = torch.mean(features, dim=1)

        # Classificação
        logits = self.classifier(aggregated_features)
        return logits

    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss, on_epoch=True)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_name == "cosine_annealing":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.eta_min)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
