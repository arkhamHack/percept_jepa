"""Model module for radar-camera fusion with V-JEPA."""

from .backbones import VJEPAEncoder, BEVRadarEncoder, ImageEncoder, RadarEncoder
from .heads import (
    ConcatFusion,
    GatedFusion,
    AnchorFreeDetectionHead,
    SpatialVelocityHead,
    TrackingEmbeddingHead,
    ObjectDecoder,
    DetectionHead,
    VelocityHead,
    TrajectoryHead,
)
from .jepa import MultimodalPerceptionModel, JEPAModel

__all__ = [
    "VJEPAEncoder",
    "BEVRadarEncoder",
    "ImageEncoder",
    "RadarEncoder",
    "ConcatFusion",
    "GatedFusion",
    "AnchorFreeDetectionHead",
    "SpatialVelocityHead",
    "TrackingEmbeddingHead",
    "ObjectDecoder",
    "DetectionHead",
    "VelocityHead",
    "TrajectoryHead",
    "MultimodalPerceptionModel",
    "JEPAModel",
]
