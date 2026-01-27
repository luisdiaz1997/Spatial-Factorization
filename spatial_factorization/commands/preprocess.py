"""Preprocess dataset into standardized format (Stage 1).

Standardizes data into common format:
- X.npy: (N, 2) spatial coordinates
- Y.npy: (N, D) count matrix (spots x genes) - ready for PNMF
- C.npy: (N,) group codes (integers 0..G-1)
- metadata.json: gene names, spot names, group names, preprocessing params
"""

import json
import time
from pathlib import Path

import numpy as np

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

    print(f"Preprocessing dataset: {config.dataset.name}")

    # Load raw data using dataset-specific loader
    data = load_dataset(config.dataset)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Create output directory for preprocessed data
    output_dir = config.output_dir / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays in standardized format
    np.save(output_dir / "X.npy", data.X.numpy())  # (N, 2)
    np.save(output_dir / "Y.npy", data.Y.numpy())  # (N, D) - ready for PNMF

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
        "dataset": config.dataset.name,
        "preprocessing": {
            "spatial_scale": config.dataset.spatial_scale,
            "filter_mt": config.dataset.filter_mt,
            "min_counts": config.dataset.min_counts,
            "min_cells": config.dataset.min_cells,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessed data saved to: {output_dir}")
    print(f"  X: {data.X.shape}")
    print(f"  Y: {data.Y.shape}")
    print(f"  C: {data.n_groups if data.n_groups > 0 else 1} groups")
