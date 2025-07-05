# Pipeline de Classificação de Vídeo com PyTorch Lightning

Este projeto implementa um pipeline completo para classificação de vídeos usando PyTorch Lightning e Hydra para configuração. Suporta modelos LSTM e Vision Transformer (ViT) para classificação de emoções em vídeos.

## Estrutura do Projeto

```
.
├── train.py                 # Script principal de treinamento
├── data_module.py          # Lightning Data Module
├── models/
│   ├── __init__.py
│   ├── lstm.py            # Modelo LSTM
│   └── vit.py             # Modelo Vision Transformer
├── config.yaml            # Configuração principal
├── requirements.txt       # Dependências
└── README.md             # Este arquivo
```

## Datasets Suportados

O pipeline suporta três datasets com a seguinte estrutura:

### CAER
```
data/caer/
├── train/
│   ├── Anger/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── val/
    ├── Anger/
    └── ...
```

### CMU-MOSEI
```
data/cmu_moisei/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── val/
└── test/
```

### RAVDESS
```
data/ravdess/
├── train/
│   ├── angry/
│   ├── calm/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
```

## Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Crie a estrutura de pastas necessária:
```bash
mkdir -p data checkpoints logs models
```

## Configuração

O arquivo `config.yaml` contém todas as configurações do pipeline. Principais parâmetros:

### Dataset
- `dataset.name`: Nome do dataset (`caer`, `cmu_moisei`, `ravdess`)
- `dataset.root`: Caminho para a pasta raiz dos dados
- `dataset.max_frames`: Número máximo de frames por vídeo
- `dataset.frames_per_second`: Taxa de extração de frames

### Modelo
- `model.name`: Tipo de modelo (`lstm`, `vit`)
- Configurações específicas para cada modelo

### Treinamento
- `training.batch_size`: Tamanho do batch
- `training.num_epochs`: Número de épocas
- `training.lr`: Taxa de aprendizado
- E outras configurações de otimização

## Uso

### Treinamento Básico
```bash
python train.py
```

### Treinamento com Configurações Específicas
```bash
# Usar dataset CAER com modelo ViT
python train.py dataset.name=caer model.name=vit

# Usar dataset RAVDESS com modelo LSTM
python train.py dataset.name=ravdess model.name=lstm

# Alterar hiperparâmetros
python train.py training.batch_size=16 training.lr=1e-3

# Combinar múltiplas configurações
python train.py dataset.name=cmu_moisei model.name=vit training.num_epochs=100
```

### Configurações Avançadas
```bash
# Usar configuração de treinamento rápido (para debugging)
python train.py training=fast

# Desabilitar early stopping
python train.py training.early_stopping.enable=false

# Alterar modelo ViT
python train.py model.name=vit model.model_name=vit_large_patch16_224
```

## Modelos Disponíveis

### LSTM
- Usa CNN (ResNet18/50) para extração de features dos frames
- LSTM bidirecional para processamento temporal
- Configurações ajustáveis de camadas e dropout

### Vision Transformer (ViT)
- Modelos pre-treinados do timm
- Diferentes métodos de agregação temporal:
  - `mean`: Média dos features
  - `max`: Máximo dos features
  - `attention`: Self-attention temporal
  - `lstm`: LSTM para agregação
  - `last`: Último frame

## Monitoramento

### TensorBoard
```bash
tensorboard --logdir=logs
```

### Métricas Registradas
- Loss de treinamento e validação
- Acurácia
- F1-Score
- Learning rate

## Estrutura de Saída

```
checkpoints/
├── modelname-datasetname-epoch=XX-val_loss=X.XX.ckpt
└── last.ckpt

logs/
└── modelname_datasetname/
    └── version_X/
        ├── events.out.tfevents...
        └── hparams.yaml
```

## Personalização

### Adicionar Novo Dataset
1. Adicione o mapeamento de classes em `data_module.py`
2. Adicione configuração no `config.yaml`
3. Organize os dados na estrutura esperada

### Adicionar Novo Modelo
1. Crie arquivo em `models/novo_modelo.py`
2. Implemente classe herda de `pl.LightningModule`
3. Adicione importação e condição em `train.py`

### Modificar Transformações
Edite as transformações em `VideoDataModule` no arquivo `data_module.py`:
- `train_transform`: Transformações de treino (com augmentação)
- `val_transform`: Transformações de validação

## Solução de Problemas

### Erro de Memória
- Reduza `training.batch_size`
- Reduza `dataset.max_frames`
- Use `training.precision=16` para mixed precision

### Dados Não Encontrados
- Verifique se `dataset.root` está correto
- Confirme se a estrutura de pastas está correta
- Verifique se há arquivos de vídeo ou imagens nas pastas

### Modelo Não Converge
- Ajuste `training.lr` (experimente valores entre 1e-5 e 1e-2)
- Verifique se os dados estão balanceados
- Considere usar `model.freeze_backbone=true` inicialmente

## Exemplos de Comandos

```bash
# Treinamento completo com ViT no CAER
python train.py dataset.name=caer model.name=vit training.num_epochs=100

# Treinamento rápido para teste
python train.py training.batch_size=4 training.num_epochs=5 training.num_workers=2

# LSTM com backbone congelado
python train.py model.name=lstm model.freeze_cnn=true

# ViT com agregação por atenção
python train.py model.name=vit model.aggregation_method=attention

# Treinamento com early stopping desabilitado
python train.py training.early_stopping.enable=false
```

## Contribuição

Para contribuir com o projeto:
1. Adicione testes para novas funcionalidades
2. Mantenha a documentação atualizada
3. Siga as convenções de código (use `black` para formatação)
4. Teste em diferentes configurações antes de submeter