"""Liver MERFISH dataset loader (healthy and diseased)."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class LiverLoader(DatasetLoader):
    """Loader for Liver MERFISH dataset.

    Supports both healthy (Cell_Type column) and diseased (Cell_Type_final column).
    Coordinates are in obsm["X_spatial"] (not the standard "spatial" key).
    Expression comes from adata.raw.X (raw counts).
    """

    DEFAULTS = {
        "spatial_scale": 50.0,
        "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_merfish.h5ad",
        "cell_type_column": "Cell_Type",
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load Liver MERFISH dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - path: str, path to h5ad file
            - cell_type_column: str, obs column for cell types
              (use "Cell_Type" for healthy, "Cell_Type_final" for diseased)
        """
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import anndata as ad
        except ImportError as e:
            raise ImportError(
                "anndata is required for LiverLoader. Install with: pip install anndata"
            ) from e

        adata = ad.read_h5ad(params["path"])
        cell_type_col = params["cell_type_column"]

        if cell_type_col not in adata.obs:
            raise RuntimeError(
                f"Missing obs['{cell_type_col}'] in {params['path']}. "
                f"Available: {list(adata.obs.columns)}"
            )

        # Extract spatial coordinates (note: X_spatial, NOT "spatial")
        X_np = np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
        X_np = rescale_spatial_coords(X_np) * params["spatial_scale"]

        # Extract gene expression from raw counts
        Y_matrix = adata.raw.X
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Extract cell type groups
        cell_types = adata.obs[cell_type_col].astype("category")
        groups_np = cell_types.cat.codes.to_numpy()
        n_groups = len(cell_types.cat.categories)
        group_names = list(cell_types.cat.categories)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        groups_t = torch.tensor(groups_np, dtype=torch.long)

        gene_names = list(adata.raw.var_names)

        return SpatialData(
            X=X_t,
            Y=Y_t,
            groups=groups_t,
            n_groups=n_groups,
            gene_names=gene_names,
            spot_names=list(adata.obs_names),
            group_names=group_names,
        )
