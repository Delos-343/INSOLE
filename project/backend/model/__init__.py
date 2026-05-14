"""
Insole Foot Classification — AI model package.

Public API:
    - build_classifier()  : Construct the multi-view multi-modal classifier.
    - load_checkpoint()   : Load weights from disk.
    - predict()           : Run inference on a triplet of images + measurements.
    - train()             : Train the model from a config.
"""

from backend.model.architectures.classifier import MultiViewFootClassifier, build_classifier
from backend.model.config import ModelConfig, TrainingConfig
from backend.model.inference.predictor import Predictor, predict
from backend.model.training.trainer import Trainer, train
from backend.model.utils.checkpoint import load_checkpoint, save_checkpoint

__all__ = [
    "MultiViewFootClassifier",
    "ModelConfig",
    "Predictor",
    "Trainer",
    "TrainingConfig",
    "build_classifier",
    "load_checkpoint",
    "predict",
    "save_checkpoint",
    "train",
]
