"""Dataset loaders for spatial transcriptomics data."""

from .base import SpatialData, DatasetLoader
from .slideseq import SlideseqLoader
from .tenxvisium import TenxVisiumLoader

LOADERS = {
    "slideseq": SlideseqLoader,
    "tenxvisium": TenxVisiumLoader,
}


def load_dataset(config) -> SpatialData:
    """Load a dataset based on configuration.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration with name and preprocessing parameters.

    Returns
    -------
    SpatialData
        Loaded and preprocessed spatial data.
    """
    if config.name not in LOADERS:
        available = ", ".join(LOADERS.keys())
        raise ValueError(f"Unknown dataset '{config.name}'. Available: {available}")

    loader_cls = LOADERS[config.name]
    return loader_cls().load(config)


__all__ = [
    "SpatialData",
    "DatasetLoader",
    "SlideseqLoader",
    "TenxVisiumLoader",
    "load_dataset",
    "LOADERS",
]
