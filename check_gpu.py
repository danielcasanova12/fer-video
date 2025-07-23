import torch
import os

print("🔥 TESTE RÁPIDO DE GPU")
print("=" * 40)

# Teste básico
cuda_available = torch.cuda.is_available()
print(f"CUDA disponível: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    
    print(f"Número de GPUs: {gpu_count}")
    print(f"GPU atual: {current_gpu}")
    print(f"Nome da GPU: {gpu_name}")
    
    # Teste de tensor
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("✅ Teste de operação: SUCESSO")
    except Exception as e:
        print(f"❌ Erro: {e}")
        
    # Memória
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Memória total: {total_memory:.1f} GB")
    
    # Configurar ambiente
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("✅ GPU configurada para uso")
    
else:
    print("❌ GPU não detectada - usando CPU")
    print("\nPara instalar PyTorch com CUDA:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# Configurar outras variáveis
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("✅ Variáveis de ambiente configuradas")
