"""Dataset loaders for spatial transcriptomics data."""

from .base import SpatialData, DatasetLoader, load_preprocessed
from .slideseq import SlideseqLoader
from .tenxvisium import TenxVisiumLoader

LOADERS = {
    "slideseq": SlideseqLoader,
    "tenxvisium": TenxVisiumLoader,
}


def load_dataset(dataset: str, preprocessing: dict = None) -> SpatialData:
    """Load a dataset by name.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'slideseq', 'tenxvisium').
    preprocessing : dict, optional
        Preprocessing parameters (spatial_scale, filter_mt, min_counts, min_cells).

    Returns
    -------
    SpatialData
        Loaded and preprocessed spatial data.
    """
    if dataset not in LOADERS:
        available = ", ".join(LOADERS.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")

    loader_cls = LOADERS[dataset]
    return loader_cls().load(preprocessing or {})


__all__ = [
    "SpatialData",
    "DatasetLoader",
    "SlideseqLoader",
    "TenxVisiumLoader",
    "load_dataset",
    "load_preprocessed",
    "LOADERS",
]
