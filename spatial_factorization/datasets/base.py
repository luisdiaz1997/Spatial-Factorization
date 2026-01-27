"""Base classes for dataset loading."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch


@dataclass
class SpatialData:
    """Container for spatial transcriptomics data.

    Attributes
    ----------
    X : torch.Tensor
        Spatial coordinates, shape (N, 2) where N is number of spots/cells.
    Y : torch.Tensor
        Count matrix, shape (N, D) where D is number of genes.
        This is the format expected by PNMF (sklearn API).
    groups : torch.Tensor, optional
        Group labels for MGGP models, shape (N,).
    n_groups : int
        Number of unique groups (0 if no groups).
    gene_names : list of str, optional
        Names of genes (length D).
    spot_names : list of str, optional
        Names of spots/cells (length N).
    group_names : list of str, optional
        Names of groups (length G).
    """

    X: torch.Tensor  # (N, 2) spatial coordinates
    Y: torch.Tensor  # (N, D) count matrix (spots x genes) - ready for PNMF
    groups: Optional[torch.Tensor] = None  # (N,) group labels
    n_groups: int = 0
    gene_names: Optional[List[str]] = field(default=None)
    spot_names: Optional[List[str]] = field(default=None)
    group_names: Optional[List[str]] = field(default=None)

    @property
    def n_spots(self) -> int:
        """Number of spots/cells."""
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.Y.shape[1]

    def to(self, device: torch.device) -> "SpatialData":
        """Move all tensors to specified device."""
        return SpatialData(
            X=self.X.to(device),
            Y=self.Y.to(device),
            groups=self.groups.to(device) if self.groups is not None else None,
            n_groups=self.n_groups,
            gene_names=self.gene_names,
            spot_names=self.spot_names,
            group_names=self.group_names,
        )

    def __repr__(self) -> str:
        return (
            f"SpatialData(n_spots={self.n_spots}, n_genes={self.n_genes}, "
            f"n_groups={self.n_groups})"
        )


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, preprocessing: dict) -> SpatialData:
        """Load and preprocess a dataset.

        Parameters
        ----------
        preprocessing : dict
            Preprocessing parameters (spatial_scale, filter_mt, min_counts, min_cells).

        Returns
        -------
        SpatialData
            Loaded and preprocessed data.
        """
        pass


def load_preprocessed(output_dir: Union[Path, str]) -> SpatialData:
    """Load preprocessed data from directory.

    After running the preprocess command, this function provides a fast path
    to load the standardized data without re-running QC filtering.

    Parameters
    ----------
    output_dir : Path or str
        Output directory containing preprocessed/ subdirectory.

    Returns
    -------
    SpatialData
        Loaded preprocessed data.
    """
    output_dir = Path(output_dir)
    prep_dir = output_dir / "preprocessed"

    if not prep_dir.exists():
        raise ValueError(f"Preprocessed data directory not found: {prep_dir}")

    # Load arrays
    X = torch.from_numpy(np.load(prep_dir / "X.npy")).float()
    Y = torch.from_numpy(np.load(prep_dir / "Y.npy")).float()
    C = torch.from_numpy(np.load(prep_dir / "C.npy")).long()

    # Load metadata
    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)

    return SpatialData(
        X=X,
        Y=Y,
        groups=C,
        n_groups=meta["n_groups"],
        gene_names=meta["gene_names"],
        spot_names=meta["spot_names"],
        group_names=meta.get("group_names"),
    )
