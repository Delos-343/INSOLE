"""Network architectures."""
from backend.model.architectures.classifier import MultiViewFootClassifier, build_classifier
from backend.model.architectures.fusion_network import MultiModalFusion
from backend.model.architectures.generative_vae import (
    ConditionalFusionVAE,
    InsoleConfigHead,
    vae_loss,
)
from backend.model.architectures.measurement_predictor import MeasurementHead
from backend.model.architectures.view_encoder import ViewEncoder

__all__ = [
    "ConditionalFusionVAE",
    "InsoleConfigHead",
    "MeasurementHead",
    "MultiModalFusion",
    "MultiViewFootClassifier",
    "ViewEncoder",
    "build_classifier",
    "vae_loss",
]
