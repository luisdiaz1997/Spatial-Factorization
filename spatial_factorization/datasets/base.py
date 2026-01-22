"""Base classes for dataset loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class SpatialData:
    """Container for spatial transcriptomics data.

    Attributes
    ----------
    X : torch.Tensor
        Spatial coordinates, shape (N, 2) where N is number of spots/cells.
    Y : torch.Tensor
        Count matrix, shape (D, N) where D is number of genes.
        Note: This is genes x spots (transposed from typical scanpy format).
    V : torch.Tensor
        Size factors for normalization, shape (N,).
    groups : torch.Tensor, optional
        Group labels for MGGP models, shape (N,).
    n_groups : int
        Number of unique groups (0 if no groups).
    gene_names : list of str, optional
        Names of genes (length D).
    spot_names : list of str, optional
        Names of spots/cells (length N).
    """

    X: torch.Tensor  # (N, 2) spatial coordinates
    Y: torch.Tensor  # (D, N) count matrix (genes x spots)
    V: torch.Tensor  # (N,) size factors
    groups: Optional[torch.Tensor] = None  # (N,) group labels
    n_groups: int = 0
    gene_names: Optional[List[str]] = field(default=None)
    spot_names: Optional[List[str]] = field(default=None)

    @property
    def n_spots(self) -> int:
        """Number of spots/cells."""
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.Y.shape[0]

    def to(self, device: torch.device) -> "SpatialData":
        """Move all tensors to specified device."""
        return SpatialData(
            X=self.X.to(device),
            Y=self.Y.to(device),
            V=self.V.to(device),
            groups=self.groups.to(device) if self.groups is not None else None,
            n_groups=self.n_groups,
            gene_names=self.gene_names,
            spot_names=self.spot_names,
        )

    def __repr__(self) -> str:
        return (
            f"SpatialData(n_spots={self.n_spots}, n_genes={self.n_genes}, "
            f"n_groups={self.n_groups})"
        )


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, config) -> SpatialData:
        """Load and preprocess a dataset.

        Parameters
        ----------
        config : DatasetConfig
            Configuration with preprocessing parameters.

        Returns
        -------
        SpatialData
            Loaded and preprocessed data.
        """
        pass
