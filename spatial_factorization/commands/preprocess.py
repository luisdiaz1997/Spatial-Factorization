"""Preprocess dataset into standardized format (Stage 1).

Standardizes data into common format:
- X.npy: (N, 2) spatial coordinates
- Y.npz: (N, D) count matrix as sparse CSR - ready for PNMF
- C.npy: (N,) group codes (integers 0..G-1)
- metadata.json: gene names, spot names, group names, preprocessing params
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from ..config import Config
from ..datasets import SpatialData, load_dataset


def _filter_nans(data: SpatialData) -> SpatialData:
    """Remove cells/spots with NaN in coordinates, expression, or groups.

    Categorical NaN codes show up as -1 (pandas convention).
    """
    import torch

    mask = np.ones(data.n_spots, dtype=bool)

    # NaN in spatial coordinates
    X_np = data.X.numpy()
    mask &= ~np.any(np.isnan(X_np), axis=1)

    # NaN in expression matrix
    Y_np = data.Y.numpy()
    mask &= ~np.any(np.isnan(Y_np), axis=1)

    # Negative group codes indicate NaN categories (pandas categorical convention)
    if data.groups is not None:
        groups_np = data.groups.numpy()
        mask &= groups_np >= 0

    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"  Removed {n_removed} cells/spots with NaN values")

    if mask.all():
        return data

    t_mask = torch.from_numpy(mask)
    idx = np.where(mask)[0]
    spot_names = (
        [data.spot_names[i] for i in idx] if data.spot_names is not None else None
    )
    return SpatialData(
        X=data.X[t_mask],
        Y=data.Y[t_mask],
        groups=data.groups[t_mask] if data.groups is not None else None,
        n_groups=data.n_groups,
        gene_names=data.gene_names,
        spot_names=spot_names,
        group_names=data.group_names,
    )


def run(config_path: str):
    """Preprocess dataset and save standardized files.

    Parameters
    ----------
    config_path : str
        Path to config YAML file.
    """
    # Load config
    config = Config.from_yaml(config_path)

    print(f"Preprocessing dataset: {config.dataset}")

    # Load raw data using dataset-specific loader
    data = load_dataset(config.dataset, config.preprocessing)
    print(f"  Loaded: {data.n_spots} spots × {data.n_genes} genes")

    # Drop any cells/spots with NaN in coordinates, expression, or group codes
    data = _filter_nans(data)
    print(f"  After NaN filter: {data.n_spots} spots × {data.n_genes} genes")

    # Create output directory for preprocessed data
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays in standardized format
    np.save(output_dir / "X.npy", data.X.numpy())  # (N, 2)

    # Save Y as sparse CSR (typically 97%+ sparse, ~60x smaller)
    Y_sparse = sparse.csr_matrix(data.Y.numpy())
    sparse.save_npz(output_dir / "Y.npz", Y_sparse)  # (N, D) sparse - ready for PNMF

    # Save group codes (C)
    if data.groups is not None:
        np.save(output_dir / "C.npy", data.groups.numpy())  # (N,)
    else:
        # No groups - create single group
        np.save(output_dir / "C.npy", np.zeros(data.n_spots, dtype=np.int64))

    # Build group names list
    group_names = None
    if data.groups is not None and data.n_groups > 0:
        # Try to get group names from data if available
        if hasattr(data, 'group_names') and data.group_names is not None:
            group_names = data.group_names
        else:
            # Fallback: create generic group names
            group_names = [f"Group_{i}" for i in range(data.n_groups)]
    else:
        group_names = ["All"]

    # Create metadata
    metadata = {
        "n_spots": data.n_spots,
        "n_genes": data.n_genes,
        "n_groups": data.n_groups if data.n_groups > 0 else 1,
        "gene_names": data.gene_names,
        "spot_names": data.spot_names,
        "group_names": group_names,
        "dataset": config.dataset,
        "preprocessing": config.preprocessing,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Report sizes
    x_size = (output_dir / "X.npy").stat().st_size / 1e6
    y_size = (output_dir / "Y.npz").stat().st_size / 1e6
    c_size = (output_dir / "C.npy").stat().st_size / 1e6
    dense_size = data.Y.numpy().nbytes / 1e6
    sparsity = 1 - (Y_sparse.nnz / Y_sparse.shape[0] / Y_sparse.shape[1])

    print(f"Preprocessed data saved to: {output_dir}")
    print(f"  X: {data.X.shape} ({x_size:.1f} MB)")
    print(f"  Y: {data.Y.shape} ({y_size:.1f} MB sparse, {dense_size:.0f} MB dense, {sparsity:.1%} sparse)")
    print(f"  C: {data.n_groups if data.n_groups > 0 else 1} groups ({c_size:.1f} MB)")
