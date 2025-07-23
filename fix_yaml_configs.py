#!/usr/bin/env python3
"""
Script para corrigir configurações em todos os arquivos YAML
"""
import os
import re
from pathlib import Path

def fix_yaml_configs():
    """Corrige configurações nos arquivos YAML"""
    src_dir = Path(__file__).parent / "src"
    yaml_files = list(src_dir.glob("*.yaml"))
    
    print(f"Encontrados {len(yaml_files)} arquivos YAML")
    
    for yaml_file in yaml_files:
        print(f"Processando: {yaml_file.name}")
        
        # Ler conteúdo
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Aplicar correções
        # 1. Precision para CPU
        content = re.sub(r'precision: 16', 'precision: 32', content)
        
        # 2. Reduzir workers para Windows
        content = re.sub(r'num_workers: [4-8]', 'num_workers: 0', content)
        
        # 3. Ajustar batch_size se muito grande
        content = re.sub(r'batch_size: (1[6-9]|[2-9]\d)', 'batch_size: 8', content)
        
        # Salvar
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ Corrigido: {yaml_file.name}")

if __name__ == "__main__":
    fix_yaml_configs()
    print("\n✅ Todas as configurações foram corrigidas!")
    print("\nPróximos passos:")
    print("1. Execute: python fix_environment.py")
    print("2. Execute: python src/train.py --config-name=frame_resnet50_caer")
