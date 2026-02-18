"""osmFISH SDMBench dataset loader."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class OsmfishLoader(DatasetLoader):
    """Loader for osmFISH SDMBench dataset (mouse somatosensory cortex).

    N=4,839 cells × D=33 genes (tiny gene panel).
    Groups: obs["ClusterName"] (~33 cell types).
    Coordinates: obsm["spatial"].
    Expression: adata.X (dense ndarray, NOT sparse).
    """

    DEFAULTS = {
        "spatial_scale": 50.0,
        "path": "/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/osmfish.h5ad",
        "group_column": "ClusterName",
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load osmFISH dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - path: str, path to h5ad file
            - group_column: str, obs column for cell types (default "ClusterName")
        """
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import scanpy as sc
        except ImportError as e:
            raise ImportError(
                "scanpy is required for OsmfishLoader. Install with: pip install scanpy"
            ) from e

        adata = sc.read_h5ad(params["path"])
        group_col = params["group_column"]

        # Extract spatial coordinates
        X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        X_np = rescale_spatial_coords(X_np) * params["spatial_scale"]

        # Expression: dense ndarray (33 genes)
        Y_matrix = adata.X
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Groups
        groups_t = None
        n_groups = 0
        group_names = None
        if group_col in adata.obs:
            clusters = adata.obs[group_col].astype("category")
            groups_np = clusters.cat.codes.to_numpy()
            groups_t = torch.tensor(groups_np, dtype=torch.long)
            n_groups = len(clusters.cat.categories)
            group_names = list(clusters.cat.categories)

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
