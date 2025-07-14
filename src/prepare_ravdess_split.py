

import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

def create_ravdess_split(ravdess_path, output_path, train_split=0.7, val_split=0.15):
    """
    Cria uma divisão estratificada do dataset RAVDESS baseada nos atores.

    Args:
        ravdess_path (str): Caminho para o diretório original do RAVDESS.
        output_path (str): Caminho para o diretório de saída da nova divisão.
        train_split (float): Proporção de atores para o conjunto de treino.
        val_split (float): Proporção de atores para o conjunto de validação.
    """
    if not os.path.exists(ravdess_path):
        print(f"Erro: O diretório do RAVDESS não foi encontrado em '{ravdess_path}'")
        return

    # Limpar e criar diretórios de saída
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    # 1. Agrupar vídeos por ator
    videos_by_actor = defaultdict(list)
    for actor_dir in sorted(os.listdir(ravdess_path)):
        if actor_dir.startswith('Actor_'):
            actor_id = int(actor_dir.split('_')[1])
            actor_path = os.path.join(ravdess_path, actor_dir)
            for filename in os.listdir(actor_path):
                if filename.endswith('.mp4'):
                    videos_by_actor[actor_id].append(os.path.join(actor_path, filename))

    actor_ids = list(videos_by_actor.keys())
    random.shuffle(actor_ids)

    # 2. Dividir atores em treino, validação e teste
    num_actors = len(actor_ids)
    train_end = int(num_actors * train_split)
    val_end = train_end + int(num_actors * val_split)

    train_actors = actor_ids[:train_end]
    val_actors = actor_ids[train_end:val_end]
    test_actors = actor_ids[val_end:]

    print(f"Total de atores: {num_actors}")
    print(f"Atores de treino: {len(train_actors)} {train_actors}")
    print(f"Atores de validação: {len(val_actors)} {val_actors}")
    print(f"Atores de teste: {len(test_actors)} {test_actors}")

    # 3. Copiar arquivos para os diretórios correspondentes
    def copy_files(actors, split_name):
        split_dir = os.path.join(output_path, split_name)
        for actor_id in tqdm(actors, desc=f"Copiando {split_name}"):
            for video_path in videos_by_actor[actor_id]:
                shutil.copy(video_path, split_dir)

    copy_files(train_actors, 'train')
    copy_files(val_actors, 'val')
    copy_files(test_actors, 'test')

    print("\nDivisão do dataset concluída com sucesso!")
    print(f"Dados salvos em: {output_path}")

if __name__ == '__main__':
    # Caminho para o dataset original. Verifique se está correto.
    original_ravdess_dir = 'data/RAVDESS' 
    # Caminho para a nova divisão estratificada
    new_split_dir = 'data/ravdess_by_actor'

    create_ravdess_split(original_ravdess_dir, new_split_dir)

