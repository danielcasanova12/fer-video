#!/usr/bin/env python3
"""
Script para configurar variáveis de ambiente e corrigir warnings
"""
import os
import warnings

def configure_environment():
    """Configura todas as variáveis de ambiente necessárias"""
    
    # Configurações do OpenMP
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "4"
    
    # Configurações do OpenCV/FFmpeg para suprimir warnings
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'
    
    # Configurações do HuggingFace
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Configurações de CUDA se disponível
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Desabilitar para performance
            print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU não detectada - usando CPU")
    except ImportError:
        print("⚠️  PyTorch não encontrado")
    
    # Configurar warnings do Python
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
    
    # Configurar OpenCV após import
    try:
        import cv2
        cv2.setLogLevel(0)  # Suprimir todos os logs do OpenCV
        print("✅ OpenCV configurado para suprimir warnings")
    except ImportError:
        print("⚠️  OpenCV não encontrado")
    
    print("✅ Todas as variáveis de ambiente configuradas!")
    
    # Mostrar configurações aplicadas
    print("\n🔧 CONFIGURAÇÕES APLICADAS:")
    print("=" * 50)
    env_vars = [
        "KMP_DUPLICATE_LIB_OK",
        "OPENCV_FFMPEG_LOGLEVEL", 
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_DISABLE_SYMLINKS_WARNING"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Não definida")
        print(f"{var}: {value}")

if __name__ == "__main__":
    configure_environment()
    
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("1. Execute: python src/train.py --config-name=frame_resnet50_caer")
    print("2. Os warnings de vídeo devem estar suprimidos agora!")
