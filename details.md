
# Detalhes do Projeto de Reconhecimento de Emoções em Vídeos

## 1. Visão Geral do Projeto

Este projeto tem como objetivo desenvolver e avaliar diferentes modelos de deep learning para a tarefa de reconhecimento de emoções faciais em vídeos (Facial Emotion Recognition - FER). A abordagem principal consiste em extrair frames de vídeos e utilizar modelos de classificação de imagem, tanto convolucionais (CNNs) quanto baseados em Transformers, para identificar a emoção presente.

## 2. Datasets

Foram utilizados três datasets públicos para treinamento e avaliação dos modelos:

### 2.1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Descrição:** Contém vídeos de 24 atores (12 homens e 12 mulheres) vocalizando duas sentenças com diferentes emoções. As emoções incluem calma, feliz, triste, com raiva, com medo, surpresa e nojo, além de uma expressão neutra.
- **Divisão do Dataset (`ravdess_by_actor`):** Para evitar o problema de **vazamento de dados** (data leakage), onde o modelo poderia aprender a reconhecer os atores em vez das emoções, foi criada uma divisão estratificada personalizada. Os 24 atores foram divididos em conjuntos de treino, validação e teste, garantindo que nenhum ator apareça em mais de um conjunto. A divisão foi feita da seguinte forma:
    - **Treino:** 70% dos atores
    - **Validação:** 15% dos atores
    - **Teste:** 15% dos atores

### 2.2. CAER (Compound Affect from Expression in the Wild)

- **Descrição:** Um dataset de vídeos "in-the-wild" (não roteirizados) que contém expressões de emoções mais naturais e complexas. As classes de emoções são similares às do RAVDESS.

### 2.3. CMU-MOSEI (CMU Multimodal Opinion Sentiment and Emotion Intensity)

- **Descrição:** Um dataset em larga escala para análise de sentimentos e emoções em vídeos. Contém milhares de vídeos do YouTube com uma ampla variedade de locutores e cenários.

## 3. Estratégia de Extração de Frames

Para representar o conteúdo temporal dos vídeos de forma eficiente, foi adotada a seguinte estratégia de amostragem de frames:

- **Número de Frames:** São extraídos **3 frames** de cada vídeo.
- **Amostragem Uniforme:** Em vez de extrair os primeiros frames, a amostragem é feita de forma uniforme ao longo do vídeo. Isso garante que os frames representem o **início, o meio e o fim** do clipe, capturando uma visão mais completa da expressão emocional.

## 4. Arquitetura dos Modelos

O projeto explora duas abordagens principais: modelos baseados em frames e modelos baseados em sequências.

### 4.1. Abordagem Baseada em Frames (`FrameClassifier`)

A arquitetura principal é modular. Um `FrameClassifier` genérico é utilizado para treinar diferentes backbones de extração de features. A lógica é a seguinte:

1.  Um backbone pré-treinado (ex: ResNet, ViT) extrai um vetor de features para cada um dos 3 frames.
2.  Os vetores de features dos frames são agregados através de uma média (`mean`).
3.  Um classificador final (uma rede neural com camadas `Linear` e `Dropout`) recebe o vetor de features agregado e prediz a emoção.

Os seguintes backbones foram utilizados:

- **ResNet50:** Uma rede neural convolucional profunda, bem estabelecida para tarefas de visão computacional.
- **ViT (Vision Transformer):** Um modelo baseado na arquitetura Transformer, que tem se mostrado muito eficaz para classificação de imagens.
- **ViTv2 (`eva02_base_patch14_224.mim_in22k`):** Uma versão mais potente e moderna do Vision Transformer, com maior capacidade de aprendizado.
- **YOLOv8l-cls:** Um modelo da família YOLO (You Only Look Once), originalmente para detecção de objetos, mas aqui utilizado como um extrator de features de imagem, aproveitando seu pré-treinamento em larga escala.

### 4.2. Abordagem Baseada em Sequências

- **LSTM (Long Short-Term Memory):** Utiliza um backbone ResNet18 para extrair features de cada frame e, em seguida, uma rede LSTM para modelar as relações temporais entre os frames e classificar a sequência.
- **Improved LSTM:** Uma versão aprimorada do modelo LSTM, que utiliza uma GRU (Gated Recurrent Unit) e um mecanismo de atenção para focar nos frames mais relevantes da sequência.

## 5. Detalhes do Treinamento

- **Frameworks:** O treinamento é orquestrado com o **PyTorch Lightning**, e as configurações são gerenciadas pelo **Hydra**, permitindo a fácil execução de múltiplos experimentos.
- **Otimizador:** Foi utilizado o otimizador **AdamW**, que é uma variação do Adam com melhor implementação do decaimento de peso (weight decay).
- **Agendador de Taxa de Aprendizagem (Scheduler):** `ReduceLROnPlateau` é utilizado para reduzir a taxa de aprendizado quando a perda de validação estabiliza.
- **Regularização:** Para combater o overfitting, foram aplicadas as seguintes técnicas:
    - **Data Augmentation:** Aumento de dados agressivo no conjunto de treino, incluindo `RandomAffine` (rotações, translações, escala) e `RandomPerspective`.
    - **Dropout:** Aplicado nas camadas finais dos classificadores, com taxas de até 0.5 para os modelos mais potentes.
    - **Weight Decay:** Utilizado para penalizar pesos grandes no otimizador.
- **Treinamento com Precisão Mista (Mixed Precision):** O treinamento é realizado com `precision=16`, o que acelera o processo e reduz o consumo de memória da GPU.

## 6. Rastreamento de Experimentos

- **Weights & Biases (W&B):** Todos os experimentos são logados no W&B. Isso inclui:
    - **Métricas:** Perda (loss), acurácia e F1-score para treino, validação e teste.
    - **Configurações:** Todos os hiperparâmetros de cada execução são salvos.
    - **Matriz de Confusão:** Ao final de cada época de treino e validação, e ao final do teste, uma matriz de confusão visual é gerada e logada no W&B para análise detalhada dos erros do modelo.
