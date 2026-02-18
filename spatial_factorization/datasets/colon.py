"""Colon Cancer Vizgen MERFISH dataset loader."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class ColonLoader(DatasetLoader):
    """Loader for Colon Cancer Vizgen MERFISH dataset.

    N=1,199,060 cells × D=492 genes.
    Groups come from an external CSV (cl46v1SubShort_ds column).
    Coordinates: obsm["spatial"] as pandas DataFrame (must sort by index).
    Expression: adata.X (no .raw).

    Due to dataset size (1.2M cells), use subsample param (e.g. subsample=10
    for ::10 subsampling to ~120K cells).
    """

    DEFAULTS = {
        "spatial_scale": 50.0,
        "h5ad_path": (
            "/gladstone/engelhardt/lab/jcai/hdp/results/merfish/"
            "HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_"
            "baysor_adata_mingenes0_mincounts10.h5ad"
        ),
        "labels_path": (
            "/gladstone/engelhardt/pelka-collaboration/"
            "Vizgen_HuColonCa_20220427_prediction-labels.csv"
        ),
        "group_column": "cl46v1SubShort_ds",
        "subsample": 10,  # take every 10th cell (~120K cells)
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load Colon Cancer MERFISH dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - h5ad_path: str, path to h5ad file
            - labels_path: str, path to labels CSV
            - group_column: str, column in labels CSV for cell types
            - subsample: int or None, step for subsampling (None = no subsampling)
        """
        params = {**self.DEFAULTS, **preprocessing}

        try:
            import anndata as ad
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "anndata and pandas are required for ColonLoader."
            ) from e

        adata = ad.read_h5ad(params["h5ad_path"])
        labels_df = pd.read_csv(params["labels_path"], index_col=0)

        # obsm["spatial"] is a DataFrame — sort cells by index
        spatial = adata.obsm["spatial"]
        if hasattr(spatial, "index"):
            order = spatial.index.argsort()
            X_raw = np.asarray(spatial.values, dtype=np.float32)[order]
        else:
            order = np.arange(len(spatial))
            X_raw = np.asarray(spatial, dtype=np.float32)[order]

        # Expression: adata.X (no .raw)
        Y_matrix = adata.X[order]
        if hasattr(Y_matrix, "toarray"):
            Y_matrix = Y_matrix.toarray()
        Y_np = np.asarray(Y_matrix, dtype=np.float32)  # (N, D)

        # Groups from external CSV (already aligned with sorted order)
        group_col = params["group_column"]
        groups_series = labels_df[group_col]

        # Apply subsampling if requested
        subsample = params.get("subsample")
        if subsample is not None and int(subsample) > 1:
            step = int(subsample)
            X_raw = X_raw[::step]
            Y_np = Y_np[::step]
            groups_series = groups_series.iloc[::step]
            print(f"  Subsampled colon to {len(X_raw):,} cells (every {step}th)")

        # Rescale coordinates
        X_np = rescale_spatial_coords(X_raw) * params["spatial_scale"]

        # Build group codes
        groups_cat = groups_series.astype("category")
        groups_np = groups_cat.cat.codes.to_numpy()
        n_groups = len(groups_cat.cat.categories)
        group_names = list(groups_cat.cat.categories)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        groups_t = torch.tensor(groups_np, dtype=torch.long)

        return SpatialData(
            X=X_t,
            Y=Y_t,
            groups=groups_t,
            n_groups=n_groups,
            gene_names=list(adata.var_names),
            spot_names=None,  # sorted indices are not meaningful cell names
            group_names=group_names,
        )
