"""SDMBench 10x Visium DLPFC dataset loader."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class SDMBenchLoader(DatasetLoader):
    """Loader for SDMBench 10x Visium DLPFC h5ad files.

    Loads single-slide Visium h5ad files from the SDMBench benchmark.
    Groups come from obs["Region"] (cortical layers L1-L6 + WM).
    Spots with NaN region (outside tissue) are filtered out.
    No QC filtering — SDMBench data is already pre-processed.
    """

    DEFAULTS = {
        "spatial_scale": 50.0,
        "path": "/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/151507.h5ad",
        "region_column": "Region",
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load SDMBench Visium DLPFC dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - path: str, path to h5ad file (e.g. .../Data/151507.h5ad)
            - region_column: str, obs column for cortical regions (default "Region")
        """
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import scanpy as sc
        except ImportError as e:
            raise ImportError(
                "scanpy is required for SDMBenchLoader. Install with: pip install scanpy"
            ) from e

        adata = sc.read_h5ad(params["path"])
        region_col = params["region_column"]

        if region_col not in adata.obs:
            raise RuntimeError(
                f"Missing obs['{region_col}'] in {params['path']}. "
                f"Available: {list(adata.obs.columns)}"
            )

        # Filter spots with NaN region (outside tissue)
        mask = ~adata.obs[region_col].isna()
        n_removed = (~mask).sum()
        if n_removed > 0:
            print(f"  Filtered {n_removed} spots with NaN region (outside tissue)")
        adata = adata[mask].copy()

        if adata.n_obs == 0:
            raise RuntimeError(f"No spots remaining after filtering NaN regions in {params['path']}")

        # Extract spatial coordinates
        X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        X_np = rescale_spatial_coords(X_np) * params["spatial_scale"]

        # Extract expression (csr_matrix, no .raw for SDMBench)
        Y_matrix = adata.X
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Extract region groups
        regions = adata.obs[region_col].astype("category")
        groups_np = regions.cat.codes.to_numpy()
        n_groups = len(regions.cat.categories)
        group_names = list(regions.cat.categories)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        groups_t = torch.tensor(groups_np, dtype=torch.long)

        return SpatialData(
            X=X_t,
            Y=Y_t,
            groups=groups_t,
            n_groups=n_groups,
            gene_names=list(adata.var_names),
            spot_names=list(adata.obs_names),
            group_names=group_names,
        )
