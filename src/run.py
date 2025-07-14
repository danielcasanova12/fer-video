import subprocess
import os
import sys

# Lista de arquivos de configuração YAML (sem a extensão .yaml)
CONFIG_LIST = [
  # "frame_res",
 #   "frame_vit",
 #  "lstm",
 #   "lstmv2",
    "frame_vitv2"
    # adicione mais aqui
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
