#!/usr/bin/env python3
"""
Script principal para treinamento de classificação de vídeo
com PyTorch Lightning e Hydra
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
import os

from data_module import VideoDataModule
from models.lstm import LSTMClassifier
from models.vit import ViTClassifier
from models.improved_lstm import ImprovedLSTMClassifier
from models.frame_classifier import FrameClassifier


@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    """Função principal de treinamento"""
    
    # Configurar seeds para reprodutibilidade
    pl.seed_everything(cfg.seed, workers=True)
    
    # Criar data module
    data_module = VideoDataModule(cfg)
    
    # Configurar data module para obter número de classes
    data_module.setup()
    num_classes = data_module.num_classes
    
    # Criar modelo baseado na configuração
    if cfg.model.model_type == "video":
        if cfg.model.name == "lstm":
            model = LSTMClassifier(
                input_size=cfg.model.input_size,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                num_classes=num_classes,
                dropout=cfg.model.dropout,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay
            )
        elif cfg.model.name == "improved_lstm":
            model = ImprovedLSTMClassifier(
                input_size=cfg.model.input_size,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                num_classes=num_classes,
                dropout=cfg.model.dropout,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay
            )
        elif cfg.model.name == "vit":
            model = ViTClassifier(
                model_name=cfg.model.model_name,
                num_classes=num_classes,
                pretrained=cfg.model.pretrained,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                freeze_backbone=cfg.model.freeze_backbone
            )
        else:
            raise ValueError(f"Modelo de vídeo não suportado: {cfg.model.name}")
    elif cfg.model.model_type == "frame":
        model = FrameClassifier(
            num_classes=num_classes,
            class_names=data_module.classes, # Adicionado para a matriz de confusão
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            cnn_backbone=cfg.model.cnn_backbone,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            scheduler_name=cfg.training.scheduler.name if "scheduler" in cfg.training and "name" in cfg.training.scheduler else None,
            t_max=cfg.training.scheduler.t_max if "scheduler" in cfg.training and "t_max" in cfg.training.scheduler else None,
            eta_min=cfg.training.scheduler.eta_min if "scheduler" in cfg.training and "eta_min" in cfg.training.scheduler else None
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {cfg.model.model_type}")
    
    # Configurar callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"{cfg.model.name}-{cfg.dataset.name}-"   "{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if cfg.training.early_stopping.enable:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping.patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Logger
    if cfg.wandb.enable:
        logger = WandbLogger(
            name=cfg.name,  # Use the name from the config file
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            log_model=True,
            config=dict(cfg)
        )
    else:
        logger = TensorBoardLogger(
            save_dir=cfg.training.log_dir,
            name=f"{cfg.model.name}_{cfg.dataset.name}",
            version=None
        )
    
    # Configurar trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        check_val_every_n_epoch=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_interval,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches
    )
    
    # Treinamento
    print(f"Iniciando treinamento do modelo {cfg.model.name} no dataset {cfg.dataset.name}")
    print(f"Número de classes: {num_classes}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.lr}")
    
    trainer.fit(model, data_module)
    
    # Teste (se disponível)
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        print("Executando avaliação no conjunto de teste...")
        trainer.test(model, data_module, ckpt_path="best")
    
    print("Treinamento concluído!")
    print(f"Melhor modelo salvo em: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()