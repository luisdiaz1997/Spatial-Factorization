"""Spatial Factorization: Dataset loaders and configs for spatial transcriptomics analysis.

This package provides:
- Dataset loaders for common spatial transcriptomics datasets
- Configuration utilities for reproducible experiments

Models and training are handled by the PNMF package:
    from PNMF import PNMF
    model = PNMF(n_components=10)
    model.fit(Y)
"""

from .config import Config
from .datasets import load_dataset, SpatialData, LOADERS

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_dataset",
    "SpatialData",
    "LOADERS",
]
