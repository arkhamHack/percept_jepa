"""Training module for radar-camera fusion models."""

from .losses import (
    jepa_latent_loss,
    detection_loss,
    velocity_loss,
    trajectory_loss,
    combined_loss,
)
from .trainer import Trainer

__all__ = [
    "jepa_latent_loss",
    "detection_loss",
    "velocity_loss",
    "trajectory_loss",
    "combined_loss",
    "Trainer",
]
