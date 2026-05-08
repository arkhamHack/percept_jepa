"""Training module for multimodal perception and legacy JEPA models."""

from .losses import (
    heatmap_focal_loss,
    box_regression_loss,
    spatial_velocity_loss,
    tracking_contrastive_loss,
    multimodal_combined_loss,
    jepa_latent_loss,
    detection_loss,
    velocity_loss,
    trajectory_loss,
    combined_loss,
)
from .trainer import Trainer

__all__ = [
    "heatmap_focal_loss",
    "box_regression_loss",
    "spatial_velocity_loss",
    "tracking_contrastive_loss",
    "multimodal_combined_loss",
    "jepa_latent_loss",
    "detection_loss",
    "velocity_loss",
    "trajectory_loss",
    "combined_loss",
    "Trainer",
]
