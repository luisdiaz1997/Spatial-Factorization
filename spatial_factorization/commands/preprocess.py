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
from ..datasets import load_dataset


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
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

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
