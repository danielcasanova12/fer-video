
"""
Módulo de modelos para classificação de vídeo
"""

from .lstm import LSTMClassifier
from .vit import ViTClassifier
from .frame_classifier import FrameClassifier

__all__ = ['LSTMClassifier', 'ViTClassifier', 'FrameClassifier']