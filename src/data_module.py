import os
import cv2
import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms, models
from omegaconf import DictConfig

# Configurar OpenCV para suprimir warnings de vídeo
cv2.setLogLevel(0)  # Suprimir todos os logs do OpenCV
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suprimir warnings do FFmpeg

class VideoDataset(Dataset):
    """Dataset para carregar frames de vídeo de estruturas de pastas."""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str,
        class_mappings: dict,
        max_frames: int = 16,
        transform: Optional[transforms.Compose] = None,
        frames_per_second: int = 1,
        model_type: str = "video" # Adicionado para diferenciar o tipo de modelo
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name  # ex: "ravdess_split"
        self.split = split                # "train", "val" ou "test"
        self.max_frames = max_frames
        self.transform = transform
        self.frames_per_second = frames_per_second

        # Suporte a nomes com "_split" e correção para CAER/CMU
        base = dataset_name
        if dataset_name == "ravdess_by_actor":
            base = "ravdess"
        elif dataset_name.endswith("_split"):
            base = dataset_name[:-len("_split")]
        elif dataset_name.upper() == "CAER":
            base = "caer"
        elif dataset_name.upper() == "CMU":
            base = "cmu_moisei"
        elif dataset_name == "cmu_moisei":
            base = "cmu_moisei"
        self.base_name = base  # ex: "ravdess"

        # Mapeamento de classes
        self.class_mappings = class_mappings
        self.classes = self.class_mappings.get(self.base_name, [])
        if not self.classes:
            raise ValueError(f"Dataset desconhecido: {self.base_name}")

        # Adicionado para suportar o novo dataset
        if self.dataset_name == 'ravdess_by_actor':
            self.base_name = 'ravdess'
            self.classes = self.class_mappings.get(self.base_name, [])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        # Carrega lista de (caminho, label)
        self.data = self._load_data()

    def _load_data(self) -> List[Tuple[str,int]]:
        data: List[Tuple[str,int]] = []
        
        # Correção para CAER que usa 'validation' ao invés de 'val'
        split_name = self.split
        if self.dataset_name.upper() == 'CAER' and self.split == 'val':
            split_name = 'validation'
            
        split_path = self.root_dir / self.dataset_name / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {split_path}")

        print(f"Procurando dados em: {split_path}")
        
        # 1) Tenta vídeos soltos (flat RAVDESS)
        exts = ['*.mp4','*.avi','*.mov','*.mkv','*.wmv']
        flat = []
        for e in exts:
            flat.extend(glob.glob(str(split_path / e)))
        
        if flat:
            print(f"Encontrados {len(flat)} vídeos soltos")
            code2emo = {
                '01':'neutral','02':'calm','03':'happy','04':'sad',
                '05':'angry','06':'fear','07':'disgust','08':'surprised'
            }
            for vf in flat:
                parts = Path(vf).stem.split('-')
                if len(parts)>=3 and parts[2] in code2emo:
                    emo = code2emo[parts[2]]
                    if emo in self.class_to_idx:
                        data.append((vf, self.class_to_idx[emo]))
            if data:
                print(f"[flat] Carregados {len(data)} vídeos de {split_path}")
                return data

        # 2) Estrutura por pastas de classe
        found_classes = []
        for cls in self.classes:
            cls_dir = split_path / cls
            if not cls_dir.exists():
                print(f"Aviso: classe '{cls}' não encontrada em {cls_dir}")
                continue

            found_classes.append(cls)
            vids = []
            for e in exts:
                vids.extend(glob.glob(str(cls_dir / e)))
            
            if vids:
                print(f"Encontrados {len(vids)} vídeos para classe '{cls}'")
                for v in vids:
                    data.append((v, self.class_to_idx[cls]))
            else:
                # subpastas com frames
                frame_folders = []
                for fld in cls_dir.iterdir():
                    if fld.is_dir():
                        imgs = list(fld.glob('*.jpg')) +  list(fld.glob('*.png'))
                        if imgs:
                            frame_folders.append(str(fld))
                            data.append((str(fld), self.class_to_idx[cls]))
                
                if frame_folders:
                    print(f"Encontradas {len(frame_folders)} pastas de frames para classe '{cls}'")
                else:
                    print(f"Nenhum dado encontrado para classe '{cls}' em {cls_dir}")

        if found_classes:
            print(f"Classes encontradas: {found_classes}")
        else:
            print("Nenhuma classe encontrada!")
            
        # 3) Se não encontrou nada, lista o conteúdo do diretório para debug
        if not data:
            print(f"\nConteúdo de {split_path}:")
            if split_path.exists():
                for item in split_path.iterdir():
                    if item.is_dir():
                        print(f"  Pasta: {item.name}")
                        # Lista alguns arquivos da pasta
                        files = list(item.glob('*'))[:5]  # Primeiros 5 arquivos
                        for f in files:
                            print(f"    {f.name}")
                        if len(list(item.glob('*'))) > 5:
                            print(f"    ... e mais {len(list(item.glob('*'))) - 5} arquivos")
                    else:
                        print(f"  Arquivo: {item.name}")
            else:
                print("  Diretório não existe!")
                
            raise ValueError(f"Nenhum dado encontrado para {self.dataset_name}/{self.split}")
            
        print(f"Total de amostras carregadas: {len(data)}")
        return data

    def _extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extrai frames de um vídeo com tratamento de erros melhorado"""
        cap = cv2.VideoCapture(video_path)
        
        # Configurar propriedades do VideoCapture para reduzir warnings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames > 0:
            # Gera índices de frames uniformemente espaçados
            indices = np.linspace(0, max(0, total_frames - 1), self.max_frames, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Verificar se o frame não está corrompido
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    # Se falhar, tenta ler alguns frames seguintes
                    for attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.shape[0] > 0:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            break

        cap.release()
        
        # Se não conseguiu extrair frames suficientes, duplica os existentes
        if len(frames) < self.max_frames and len(frames) > 0:
            while len(frames) < self.max_frames:
                frames.append(frames[-1])  # Duplica o último frame válido
        
        return frames

    def _load_frames_from_folder(self, folder: str) -> List[np.ndarray]:
        files = sorted(glob.glob(os.path.join(folder,'*.jpg')) +
                       glob.glob(os.path.join(folder,'*.png')))
        if not files:
            print(f"Nenhum frame encontrado em: {folder}")
            return []
            
        step = max(1, len(files) // self.max_frames)
        frames = []
        for i in range(0, len(files), step):
            if len(frames) >= self.max_frames: 
                break
            img = cv2.imread(files[i])
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,int]:
        path, label = self.data[idx]
        
        if os.path.isfile(path):
            frames = self._extract_frames_from_video(path)
        else:
            frames = self._load_frames_from_folder(path)

        if not frames:
            print(f"Aviso: nenhum frame carregado para {path}")
            frames = [np.zeros((224,224,3), dtype=np.uint8)]

        # pad ou truncate
        if len(frames) < self.max_frames:
            # Replica o último frame
            frames.extend([frames[-1]] * (self.max_frames - len(frames)))
        else:
            frames = frames[:self.max_frames]

        # transforma
        if self.transform:
            out = [self.transform(Image.fromarray(f)) for f in frames]
        else:
            default_tf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            out = [default_tf(Image.fromarray(f)) for f in frames]

        # T x C x H x W
        return torch.stack(out), label


class VideoDataModule(pl.LightningDataModule):
    """Lightning DataModule para PyTorch Lightning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg.dataset.root
        self.dataset = cfg.dataset.name
        self.max_frames = cfg.dataset.max_frames
        self.fps = cfg.dataset.frames_per_second
        self.batch = cfg.training.batch_size
        self.workers = cfg.training.num_workers
        self.class_mappings = cfg.dataset.class_mappings
        self.model_type = cfg.model.model_type # Adicionado

        # Determinar número de classes baseado no dataset
        base_name = self.dataset
        if self.dataset == "ravdess_by_actor":
            base_name = "ravdess"
        elif self.dataset.endswith("_split"):
            base_name = self.dataset[:-len("_split")]
        elif self.dataset == "cmu-moisei_split":
            base_name = "cmu_moisei"
        elif self.dataset.upper() == "CAER":
            base_name = "caer"
        elif self.dataset.upper() == "CMU":
            base_name = "cmu_moisei"
        elif self.dataset == "cmu_moisei":
            base_name = "cmu_moisei"
        
        self.classes = self.class_mappings.get(base_name, [])
        if not self.classes:
            raise ValueError(f"Dataset desconhecido: {base_name}. Datasets disponíveis: {list(self.class_mappings.keys())}")
        
        self.num_classes = len(self.classes)

        # transforms
        self.train_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def setup(self, stage: Optional[str]=None):
        print(f"Configurando datasets para: {self.dataset}")
        print(f"Diretório raiz: {self.root_dir}")
        
        self.train_ds = VideoDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset,
            split='train',
            class_mappings=self.class_mappings,
            max_frames=self.max_frames,
            transform=self.train_tf,
            frames_per_second=self.fps,
            model_type=self.model_type
        )
        self.val_ds = VideoDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset,
            split='val',
            class_mappings=self.class_mappings,
            max_frames=self.max_frames,
            transform=self.val_tf,
            frames_per_second=self.fps,
            model_type=self.model_type
        )
        
        test_path = Path(self.root_dir) / self.dataset / 'test'
        if test_path.exists():
            self.test_ds = VideoDataset(
                root_dir=self.root_dir,
                dataset_name=self.dataset,
                split='test',
                class_mappings=self.class_mappings,
                max_frames=self.max_frames,
                transform=self.val_tf,
                frames_per_second=self.fps,
                model_type=self.model_type
            )
        else:
            print(f"Conjunto de teste não encontrado em: {test_path}")
            self.test_ds = None

    def train_dataloader(self) -> DataLoader:
        # Detectar se estamos usando GPU ou CPU
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.train_ds, batch_size=self.batch, shuffle=True,
            num_workers=self.workers, pin_memory=pin_memory,
            persistent_workers=self.workers>0
        )

    def val_dataloader(self) -> DataLoader:
        # Detectar se estamos usando GPU ou CPU
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.val_ds, batch_size=self.batch, shuffle=False,
            num_workers=self.workers, pin_memory=pin_memory,
            persistent_workers=self.workers>0
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds:
            # Detectar se estamos usando GPU ou CPU
            pin_memory = torch.cuda.is_available()
            
            return DataLoader(
                self.test_ds, batch_size=self.batch, shuffle=False,
                num_workers=self.workers, pin_memory=pin_memory,
                persistent_workers=self.workers>0
            )
        return None