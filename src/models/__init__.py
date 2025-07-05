
"""
Módulo de modelos para classificação de vídeo
"""

from .lstm import LSTMClassifier
from .vit import ViTClassifier

__all__ = ['LSTMClassifier', 'ViTClassifier']