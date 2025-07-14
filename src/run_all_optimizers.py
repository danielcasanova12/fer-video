import subprocess
import os
import sys

# --- Configurações ---
MODELS_TO_OPTIMIZE = [
    "frame_res",
    "frame_vit",
]

# --- Diretório Base ---
# Obtém o diretório do script atual (src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_SCRIPT = os.path.join(SRC_DIR, "optimizer", "optimizer.py")

# --- Executar Otimização para Cada Modelo ---
def main():
    for model_config_name in MODELS_TO_OPTIMIZE:
        print(f"\n--- Otimizando para o modelo: {model_config_name} ---\n")
        
        command = [
            sys.executable,  # Usa o mesmo interpretador python
            OPTIMIZER_SCRIPT,
            f"model_config={model_config_name}"
        ]
        
        try:
            # Executa o comando a partir da pasta src, passando o ambiente completo
            subprocess.run(command, check=True, cwd=SRC_DIR, env=os.environ.copy())
            print(f"\n--- Otimização para {model_config_name} concluída com sucesso! ---\n")
        except subprocess.CalledProcessError as e:
            print(f"\n--- Erro ao otimizar {model_config_name}: {e} ---\n")

if __name__ == "__main__":
    main()