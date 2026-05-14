"""Configuration dataclasses for the foot-classification model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------
# 5-class scheme drawn directly from the project brief:
#   Normal Foot, Flat Arch, Severe Flat Arch, High Arch, Severe High Arch
# Driven by arch height (cm) bands:
#   NORMAL          : 3.6 – 5.5 cm
#   FLAT ARCH       : 2.7 – 3.5 cm
#   SEVERE FLAT     : < 2.7 cm
#   HIGH ARCH       : 5.6 – 6.4 cm
#   SEVERE HIGH     : > 6.4 cm
CLASS_NAMES: tuple[str, ...] = (
    "Severe Flat Arch",
    "Flat Arch",
    "Normal Foot",
    "High Arch",
    "Severe High Arch",
)

# Arch-height (cm) decision bands. Used by the rule-based labeller (data prep)
# and by the GUI to render a coloured severity meter.
ARCH_HEIGHT_BANDS: dict[str, tuple[float, float]] = {
    "Severe Flat Arch": (0.0, 2.7),
    "Flat Arch":         (2.7, 3.6),
    "Normal Foot":       (3.6, 5.6),
    "High Arch":         (5.6, 6.5),
    "Severe High Arch":  (6.5, 99.0),
}

# Heel angle (deg) bands — valgus / varus alignment, per brief:
#   NORMAL: 0–5°,  VALGUS(FLAT): >5°,  VALGUS(SEVERE FLAT): >10°,
#   VARUS(HIGH): <0°,  VARUS(SEVERE HIGH): <-5°
HEEL_ANGLE_BANDS: dict[str, tuple[float, float]] = {
    "Severe Flat Arch": (10.0, 90.0),
    "Flat Arch":         (5.0, 10.0),
    "Normal Foot":       (0.0, 5.0),
    "High Arch":         (-5.0, 0.0),
    "Severe High Arch":  (-90.0, -5.0),
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Hyper-parameters describing the architecture (not training)."""

    # Backbones (per view). We use timm-compatible names so swapping is trivial.
    lateral_backbone: str = "efficientnet_b0"
    top_backbone: str = "efficientnet_b0"
    back_backbone: str = "efficientnet_b0"

    # Whether each backbone is initialised with ImageNet weights.
    pretrained: bool = True

    # Embedding dimension per view backbone (output of global pool).
    view_embed_dim: int = 1280  # EfficientNet-B0 default; auto-detected at build.

    # Dimension of the fusion-bottleneck representation.
    fusion_dim: int = 512

    # Number of scalar measurement features (calcaneal incl., heel angle,
    # arch height, kite angle, 1st metatarsal-talus angle).
    num_measurements: int = 5

    # Number of output classes.
    num_classes: int = len(CLASS_NAMES)

    # Dropout on the classification head.
    head_dropout: float = 0.30

    # Whether to include a generative VAE branch that learns a latent
    # representation of the input — used for both augmentation and
    # "insole pattern" generation.
    use_generative_branch: bool = True
    vae_latent_dim: int = 128

    # Whether to attach a measurement-prediction regression head (multi-task).
    # Lets the network predict the 5 angles directly from images during
    # inference when the user hasn't provided manual measurements.
    predict_measurements: bool = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Hyper-parameters describing training (optimisation)."""

    # Data
    data_dir: Path = Path("data")
    sheet_path: Path | None = None   # Resolved from data_dir/Sheet/*.xlsx if None
    image_size: int = 256
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    random_state: int = 42

    # Optimisation
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    scheduler: Literal["cosine", "step", "plateau"] = "cosine"

    # Loss weighting (only used when multi-task heads are active)
    cls_loss_weight: float = 1.0
    measurement_loss_weight: float = 0.5
    vae_loss_weight: float = 0.1

    # Hardware
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True

    # Augmentation
    use_augmentation: bool = True

    # Checkpointing / logging
    output_dir: Path = Path("backend/model/checkpoints")
    save_every: int = 5
    keep_top_k: int = 3
    log_every: int = 10
    early_stopping_patience: int = 10

    # Targets from the brief.
    target_accuracy: float = 0.90


@dataclass
class InferenceConfig:
    """Settings used at prediction time."""

    checkpoint_path: Path = Path("backend/model/checkpoints/foot_classifier_v1.pt")
    image_size: int = 256
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    use_tta: bool = False                 # Test-time augmentation
    confidence_threshold: float = 0.0     # Below this we return "uncertain"
    class_names: tuple[str, ...] = field(default_factory=lambda: CLASS_NAMES)
