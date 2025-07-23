#!/usr/bin/env python3
"""
Script launcher que configura ambiente e executa treinamento
"""
import os
import sys
import warnings
import subprocess

def setup_environment():
    """Configura ambiente antes de executar qualquer coisa"""
    
    # Configurações críticas ANTES de qualquer import
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Suprimir warnings do Python
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print("🔧 Ambiente configurado!")

def run_training(config_name="frame_resnet50_caer"):
    """Executa o treinamento com configurações otimizadas"""
    
    setup_environment()
    
    # Importar após configurar ambiente
    try:
        import cv2
        cv2.setLogLevel(0)
    except ImportError:
        pass
    
    print(f"🚀 Iniciando treinamento com {config_name}")
    
    # Executar treinamento
    cmd = [
        sys.executable,
        "src/train.py", 
        f"--config-name={config_name}"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", check=True)
        print("✅ Treinamento concluído com sucesso!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no treinamento: {e}")
        return False

if __name__ == "__main__":
    # Pegar nome da configuração se fornecido
    config = sys.argv[1] if len(sys.argv) > 1 else "frame_resnet50_caer"
    
    print("🎯 FER-VIDEO LAUNCHER")
    print("=" * 50)
    
    success = run_training(config)
    
    if success:
        print("\n🎉 TREINAMENTO FINALIZADO!")
    else:
        print("\n💥 ERRO NO TREINAMENTO!")
        
    print("=" * 50)
