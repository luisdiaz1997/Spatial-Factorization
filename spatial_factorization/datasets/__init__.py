"""Dataset loaders for spatial transcriptomics data."""

from .base import SpatialData, DatasetLoader, load_preprocessed
from .slideseq import SlideseqLoader
from .tenxvisium import TenxVisiumLoader
from .sdmbench import SDMBenchLoader
from .liver import LiverLoader
from .merfish import MerfishLoader
from .colon import ColonLoader
from .osmfish import OsmfishLoader

LOADERS = {
    "slideseq": SlideseqLoader,
    "tenxvisium": TenxVisiumLoader,
    "sdmbench": SDMBenchLoader,
    "liver": LiverLoader,
    "merfish": MerfishLoader,
    "colon": ColonLoader,
    "osmfish": OsmfishLoader,
}


def load_dataset(dataset: str, preprocessing: dict = None) -> SpatialData:
    """Load a dataset by name.

    Parameters
    ----------
    dataset : str
        Dataset name. Available: slideseq, tenxvisium, sdmbench, liver,
        merfish, colon, osmfish.
    preprocessing : dict, optional
        Preprocessing parameters (dataset-specific).

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
    "SDMBenchLoader",
    "LiverLoader",
    "MerfishLoader",
    "ColonLoader",
    "OsmfishLoader",
    "load_dataset",
    "load_preprocessed",
    "LOADERS",
]
