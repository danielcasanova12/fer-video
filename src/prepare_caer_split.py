

import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

def create_caer_split(caer_path, output_path, train_split=0.7, val_split=0.15):
    """
    Cria uma divisão estratificada do dataset CAER.

    Args:
        caer_path (str): Caminho para o diretório original do CAER.
        output_path (str): Caminho para o diretório de saída da nova divisão.
        train_split (float): Proporção para o conjunto de treino.
        val_split (float): Proporção para o conjunto de validação.
    """
    if not os.path.exists(caer_path):
        print(f"Erro: O diretório do CAER não foi encontrado em '{caer_path}'")
        return

    # Limpar e criar diretórios de saída
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    # Agrupar vídeos por classe
    videos_by_class = defaultdict(list)
    for class_dir in sorted(os.listdir(caer_path)):
        class_path = os.path.join(caer_path, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.avi'): # Assumindo que os vídeos são .avi
                    videos_by_class[class_dir].append(os.path.join(class_path, filename))

    all_videos = []
    for cls, videos in videos_by_class.items():
        for video_path in videos:
            all_videos.append((video_path, cls))

    random.shuffle(all_videos)

    # Dividir vídeos em treino, validação e teste
    num_videos = len(all_videos)
    train_end = int(num_videos * train_split)
    val_end = train_end + int(num_videos * val_split)

    train_videos = all_videos[:train_end]
    val_videos = all_videos[train_end:val_end]
    test_videos = all_videos[val_end:]

    print(f"Total de vídeos: {num_videos}")
    print(f"Vídeos de treino: {len(train_videos)}")
    print(f"Vídeos de validação: {len(val_videos)}")
    print(f"Vídeos de teste: {len(test_videos)}")

    # Copiar arquivos para os diretórios correspondentes
    def copy_files(videos, split_name):
        for video_path, cls in tqdm(videos, desc=f"Copiando {split_name}"):
            dest_dir = os.path.join(output_path, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(video_path, dest_dir)

    copy_files(train_videos, 'train')
    copy_files(val_videos, 'val')
    copy_files(test_videos, 'test')

    print("\nDivisão do dataset CAER concluída com sucesso!")
    print(f"Dados salvos em: {output_path}")

if __name__ == '__main__':
    original_caer_dir = 'data/CAER' # Verifique se este é o caminho correto
    new_split_dir = 'data/caer_split'

    create_caer_split(original_caer_dir, new_split_dir)

