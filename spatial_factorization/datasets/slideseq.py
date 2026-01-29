"""SlideseqV2 dataset loader."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData


def rescale_spatial_coords(X: np.ndarray) -> np.ndarray:
    """Rescale spatial coordinates to [0, 1] range."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)


class SlideseqLoader(DatasetLoader):
    """Loader for SlideseqV2 dataset from squidpy."""

    # Default preprocessing parameters
    DEFAULTS = {
        "spatial_scale": 50.0,
        "filter_mt": True,
        "min_counts": 100,
        "min_cells": 10,
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load SlideseqV2 dataset.

        Parameters
        ----------
        preprocessing : dict
            Preprocessing parameters:
            - spatial_scale: float, coordinate scaling factor (default: 50.0)
            - filter_mt: bool, whether to filter mitochondrial genes (default: True)
            - min_counts: int, minimum counts per cell (default: 100)
            - min_cells: int, minimum cells per gene (default: 10)

        Returns
        -------
        SpatialData
            Loaded dataset.
        """
        # Merge with defaults
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import scanpy as sc
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy and scanpy are required for SlideseqV2. "
                "Install with: pip install squidpy scanpy"
            ) from e

        # Load dataset
        adata = sq.datasets.slideseqv2()
        adata = adata.raw.to_adata()

        # Mark mitochondrial genes
        adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
        adata.var["MT"] = adata.var["mt"]

        # QC filtering
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs.pct_counts_mt < 20].copy()
        sc.pp.filter_cells(adata, min_counts=params["min_counts"])
        sc.pp.filter_genes(adata, min_cells=params["min_cells"])

        # Filter MT genes if requested
        if params["filter_mt"]:
            gene_mask = ~adata.var["MT"].values
            adata = adata[:, gene_mask]

        # Extract spatial coordinates
        X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        X_np = rescale_spatial_coords(X_np) * params["spatial_scale"]

        # Extract count matrix (N x D format for PNMF)
        Y_matrix = adata.X
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Extract groups if available (for MGGP)
        groups_t = None
        n_groups = 0
        group_names = None
        if "cluster" in adata.obs:
            clusters = adata.obs["cluster"].astype("category")
            cluster_codes = clusters.cat.codes.to_numpy()
            groups_t = torch.tensor(cluster_codes, dtype=torch.long)
            n_groups = len(clusters.cat.categories)
            group_names = list(clusters.cat.categories)

        # Convert to tensors
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)

        return SpatialData(
            X=X_t,
            Y=Y_t,
            groups=groups_t,
            n_groups=n_groups,
            gene_names=list(adata.var_names),
            spot_names=list(adata.obs_names),
            group_names=group_names,
        )
