
import optuna
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger

import wandb
import sys
import os
import hydra.utils

# Adiciona a pasta src ao sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_module import VideoDataModule
from models.frame_classifier import FrameClassifier
from models.vit import ViTClassifier

def objective(trial: optuna.Trial, config: DictConfig):
    """
    Optuna objective function.
    """
    # --- Hyperparameter search space ---
    config.training.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config.training.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    config.training.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Convert config to a native Python dict to ensure native types are passed
    # This resolves the 'ConfigAttributeError: Key 'startswith' is not in struct'
    resolved_config = OmegaConf.to_container(config, resolve=True)

    # --- DataModule ---
    datamodule = VideoDataModule(config)
    datamodule.setup() # Setup datamodule to get num_classes
    num_classes = datamodule.num_classes

    # --- Model ---
    if resolved_config['model']['model_type'] == "frame":
        model = FrameClassifier(
            num_classes=num_classes,
            lr=resolved_config['training']['lr'],
            weight_decay=resolved_config['training']['weight_decay'],
            cnn_backbone=resolved_config['model']['cnn_backbone'],
            pretrained=resolved_config['model']['pretrained'],
            freeze_backbone=resolved_config['model']['freeze_backbone']
        )
    elif resolved_config['model']['model_type'] == "vit":
        model = ViTClassifier(
            num_classes=num_classes,
            lr=resolved_config['training']['lr'],
            weight_decay=resolved_config['training']['weight_decay'],
            model_name=resolved_config['model']['cnn_backbone'], # For ViT, cnn_backbone is the model_name
            pretrained=resolved_config['model']['pretrained'],
            freeze_backbone=resolved_config['model']['freeze_backbone']
        )
    else:
        raise ValueError(f"Unknown model type: {resolved_config['model']['model_type']}")

    # --- Trainer ---
    wandb_logger = WandbLogger(
        project=resolved_config['wandb']['project'],
        entity=resolved_config['wandb']['entity'],
        name=f"{resolved_config['name']}-{trial.number}",
        config=resolved_config
    )

    trainer = Trainer(
        max_epochs=resolved_config['training']['num_epochs'],
        logger=wandb_logger,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=resolved_config['training']['precision'],
        callbacks=[] # Removido o PyTorchLightningPruningCallback
    )

    # --- Training ---
    # Adiciona o callback de pruning manual
    class OptunaPruningCallback(Callback):
        def __init__(self, trial, monitor):
            self.trial = trial
            self.monitor = monitor

        def on_validation_end(self, trainer, pl_module):
            logs = trainer.callback_metrics
            current_score = logs.get(self.monitor)
            if current_score is None:
                return
            self.trial.report(current_score, trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    trainer.callbacks.append(OptunaPruningCallback(trial, monitor="val_loss"))

    trainer.fit(model, datamodule)

    # --- Cleanup ---
    wandb.finish()

    return trainer.callback_metrics["val_loss"].item()

@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Constrói o caminho absoluto para o arquivo de config do modelo
    original_cwd = hydra.utils.get_original_cwd()
    # Ajusta o caminho para procurar na pasta src
    model_config_path = os.path.join(original_cwd, f"{cfg.model_config}.yaml")
    model_config = OmegaConf.load(model_config_path)
    
    # Merge com a configuração base
    config = OmegaConf.merge(cfg, model_config)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, config), n_trials=20)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
