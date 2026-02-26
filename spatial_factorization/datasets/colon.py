"""Colon Cancer Vizgen MERFISH dataset loader."""

from __future__ import annotations

import numpy as np
import torch

from .base import DatasetLoader, SpatialData
from .slideseq import rescale_spatial_coords


class ColonLoader(DatasetLoader):
    """Loader for Colon Cancer Vizgen MERFISH dataset.

    N=1,199,060 cells × D=492 genes.
    Groups can come from an external CSV or directly from adata.obs.
    Coordinates: obsm["spatial"] as pandas DataFrame, or obs["center_x"]/["center_y"].
    Expression: adata.X (no .raw).

    Full dataset: 1.2M cells. Use subsample param (int step) to downsample if needed.

    When labels_path is None (or omitted), groups are read from adata.obs[group_column].
    Coordinate detection:
      - If obs["center_x"] exists → use obs["center_x"], obs["center_y"] (Patient 1 sisi format)
      - Otherwise → use obsm["spatial"] with DataFrame-sort logic
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
        "subsample": None,  # no subsampling by default
    }

    def load(self, preprocessing: dict) -> SpatialData:
        """Load Colon Cancer MERFISH dataset.

        Parameters
        ----------
        preprocessing : dict
            - spatial_scale: float (default 50.0)
            - h5ad_path: str, path to h5ad file
            - labels_path: str or None, path to labels CSV (None = read from adata.obs)
            - group_column: str, column in labels CSV or adata.obs for cell types
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

        # --- Coordinate extraction ---
        if "center_x" in adata.obs.columns:
            # Patient 1 sisi_ingest format: coords in obs columns
            X_raw = np.stack(
                [
                    adata.obs["center_x"].to_numpy(dtype=np.float32),
                    adata.obs["center_y"].to_numpy(dtype=np.float32),
                ],
                axis=1,
            )
            order = np.arange(len(adata))
        else:
            # obsm["spatial"] may be a DataFrame — sort cells by index
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

        # --- Groups ---
        group_col = params["group_column"]
        labels_path = params.get("labels_path")

        if labels_path is None:
            # Inline labels: read directly from adata.obs (already aligned with order)
            groups_series = adata.obs[group_col].iloc[order].reset_index(drop=True)
        else:
            # External CSV (already aligned with sorted order)
            import pandas as pd  # noqa: F811 — may already be imported above
            labels_df = pd.read_csv(labels_path, index_col=0)
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
