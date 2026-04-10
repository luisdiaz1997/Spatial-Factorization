"""squidpy MERFISH dataset loader (Moffitt et al. mouse hypothalamus, 2D)."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class MerfishLoader(DatasetLoader):
    """Loader for squidpy MERFISH dataset (mouse hypothalamus, 2D spatial).

    Uses sq.datasets.merfish() which downloads automatically.
    N=73,655 cells × D=161 genes.
    Groups: obs["Cell_class"] (16 cell classes).
    Coordinates: obsm["spatial"] (Centroid_X, Centroid_Y) — 2D.
    Expression: adata.X (sparse, no .raw).
    """

    DEFAULTS = {
        "spatial_scale": 50.0,
        "group_column": "Cell_class",
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load squidpy MERFISH dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - group_column: str, obs column for cell classes (default "Cell_class")
        """
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy is required for MerfishLoader. Install with: pip install squidpy"
            ) from e

        adata = sq.datasets.merfish()
        group_col = params["group_column"]

        # Use 2D spatial coords
        X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        X_np = rescale_spatial_coords(X_np) * params["spatial_scale"]

        # Expression: adata.X (sparse, no .raw)
        Y_matrix = adata.X
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Groups
        groups_t = None
        n_groups = 0
        group_names = None
        if group_col in adata.obs:
            cell_classes = adata.obs[group_col].astype("category")
            groups_np = cell_classes.cat.codes.to_numpy()
            groups_t = torch.tensor(groups_np, dtype=torch.long)
            n_groups = len(cell_classes.cat.categories)
            group_names = list(cell_classes.cat.categories)

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
