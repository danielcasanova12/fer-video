#!/usr/bin/env python3
"""
Script para testar e verificar GPU availability
"""
import torch
import os
import sys

def test_gpu():
    """Testa disponibilidade e configuração da GPU"""
    print("=" * 60)
    print("🔍 VERIFICAÇÃO DE GPU")
    print("=" * 60)
    
    # 1. Verificar se CUDA está disponível
    print(f"1. CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 2. Informações da GPU
        print(f"2. Número de GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Memória
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   Memória total: {total_memory:.1f} GB")
            
            # Memória livre
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = total_memory - allocated
            
            print(f"   Memória alocada: {allocated:.1f} GB")
            print(f"   Memória cached: {cached:.1f} GB") 
            print(f"   Memória livre: {free:.1f} GB")
        
        # 3. Testar operação na GPU
        print(f"\n3. GPU atual: {torch.cuda.current_device()}")
        
        try:
            # Criar tensor na GPU
            device = torch.device('cuda')
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.mm(test_tensor, test_tensor)
            print("✅ Teste de operação na GPU: SUCESSO")
            
            # Verificar se PyTorch Lightning detecta GPU
            try:
                import pytorch_lightning as pl
                from pytorch_lightning.accelerators import find_usable_cuda_devices
                usable_gpus = find_usable_cuda_devices()
                print(f"✅ PyTorch Lightning detectou GPUs: {usable_gpus}")
            except Exception as e:
                print(f"⚠️  Erro ao verificar PyTorch Lightning: {e}")
                
        except Exception as e:
            print(f"❌ Erro no teste da GPU: {e}")
    
    else:
        print("❌ CUDA não está disponível!")
        print("\nPossíveis causas:")
        print("- Drivers NVIDIA não instalados")
        print("- PyTorch instalado sem suporte CUDA")
        print("- GPU não compatível")
        
        # Verificar versão do PyTorch
        print(f"\nVersão do PyTorch: {torch.__version__}")
        print(f"Versão CUDA compilada: {torch.version.cuda}")
        
    print("\n" + "=" * 60)
    
    # 4. Configurar variáveis de ambiente
    print("🔧 CONFIGURAÇÃO DE AMBIENTE")
    print("=" * 60)
    
    if torch.cuda.is_available():
        # Configurar para usar GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Para debug
        print("✅ Configurado para usar GPU 0")
    else:
        print("⚠️  Configurado para usar CPU")
    
    # Configurar outras variáveis
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    print("✅ Variáveis de ambiente configuradas")
    
    return torch.cuda.is_available()

def create_gpu_config():
    """Cria configuração otimizada para GPU"""
    gpu_available = torch.cuda.is_available()
    
    config = {
        "precision": "16-mixed" if gpu_available else "32",
        "accelerator": "gpu" if gpu_available else "cpu",
        "devices": 1 if gpu_available else "auto",
        "num_workers": 4 if gpu_available else 0,
        "batch_size": 32 if gpu_available else 8,
        "pin_memory": gpu_available
    }
    
    print("\n🚀 CONFIGURAÇÃO RECOMENDADA")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    return config

def update_yaml_for_gpu():
    """Atualiza arquivos YAML para usar configuração de GPU"""
    from pathlib import Path
    import re
    
    gpu_available = torch.cuda.is_available()
    src_dir = Path("src")
    
    if not src_dir.exists():
        print("⚠️  Pasta 'src' não encontrada")
        return
    
    yaml_files = list(src_dir.glob("*.yaml"))
    
    print(f"\n📝 ATUALIZANDO {len(yaml_files)} ARQUIVOS YAML")
    print("=" * 60)
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if gpu_available:
                # Configuração para GPU
                content = re.sub(r'precision: \d+', 'precision: "16-mixed"', content)
                content = re.sub(r'num_workers: \d+', 'num_workers: 4', content)
                content = re.sub(r'batch_size: \d+', 'batch_size: 32', content)
            else:
                # Configuração para CPU
                content = re.sub(r'precision: .*', 'precision: "32"', content)
                content = re.sub(r'num_workers: \d+', 'num_workers: 0', content)
                content = re.sub(r'batch_size: \d+', 'batch_size: 8', content)
            
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ {yaml_file.name}")
            
        except Exception as e:
            print(f"❌ Erro em {yaml_file.name}: {e}")

if __name__ == "__main__":
    # Teste completo
    gpu_available = test_gpu()
    config = create_gpu_config()
    
    print(f"\n🎯 RESULTADO FINAL")
    print("=" * 60)
    if gpu_available:
        print("✅ GPU DETECTADA E CONFIGURADA!")
        print("🚀 Projeto otimizado para treinamento em GPU")
        
        # Atualizar configurações
        update_yaml_for_gpu()
        
        print("\n📋 PRÓXIMOS PASSOS:")
        print("1. Execute: python src/train.py --config-name=frame_resnet50_caer")
        print("2. Monitore o uso da GPU com: nvidia-smi")
        
    else:
        print("⚠️  GPU NÃO DETECTADA")
        print("🔧 Projeto configurado para CPU")
        print("\n📋 PARA HABILITAR GPU:")
        print("1. Instale CUDA Toolkit")
        print("2. Reinstale PyTorch com CUDA:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 60)
