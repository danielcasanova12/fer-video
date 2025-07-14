import subprocess
import os
import sys

# Lista de arquivos de configuração YAML (sem a extensão .yaml)
CONFIG_LIST = [
    # ResNet50
    "frame_resnet50_ravdess_by_actor",
    "frame_resnet50_caer",
    "frame_resnet50_cmu_moisei",
    # ViT
    "frame_vit_ravdess_by_actor",
    "frame_vit_caer",
    "frame_vit_cmu_moisei",
    # ViTv2
    "frame_vitv2_ravdess_by_actor",
    "frame_vitv2_caer",
    "frame_vitv2_cmu_moisei",
    # LSTM
    "lstm_ravdess_by_actor",
    "lstm_caer",
    "lstm_cmu_moisei",
    # Improved LSTM
    "improved_lstm_ravdess_by_actor",
    "improved_lstm_caer",
    "improved_lstm_cmu_moisei",
    # YOLOv8
    "frame_yolov8l_cls_ravdess_by_actor",
    "frame_yolov8l_cls_caer",
    "frame_yolov8l_cls_cmu_moisei",
]

# Caminho para o script principal
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório atual
TRAIN_SCRIPT = os.path.join(SRC_DIR, "train.py")  # Nome do seu script principal

def main():
    for config_name in CONFIG_LIST:
        print(f"\n--- Iniciando treino com configuração: {config_name}.yaml ---\n")

        command = [
            sys.executable,  # Usa o mesmo Python atual
            TRAIN_SCRIPT,
            f"--config-name={config_name}"
        ]

        try:
            subprocess.run(command, check=True, cwd=SRC_DIR, env=os.environ.copy())
            print(f"\n--- Treinamento com {config_name} finalizado com sucesso! ---\n")
        except subprocess.CalledProcessError as e:
            print(f"\n--- Erro ao treinar com {config_name}: {e} ---\n")

if __name__ == "__main__":
    main()
