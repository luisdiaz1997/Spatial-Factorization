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

    # Group names that are None or float NaN (NaN present as an explicit category)
    if data.groups is not None and data.group_names is not None:
        import math
        nan_name_codes = [
            i for i, name in enumerate(data.group_names)
            if name is None or (isinstance(name, float) and math.isnan(name))
        ]
        if nan_name_codes:
            codes_np = data.groups.numpy()
            mask &= ~np.isin(codes_np, nan_name_codes)
            print(f"  Found {len(nan_name_codes)} group(s) with None/NaN name")

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


def _filter_small_groups(data: SpatialData, min_group_size: int) -> SpatialData:
    """Remove cells belonging to groups smaller than min_group_size, then re-encode codes.

    After dropping small groups, codes are remapped to contiguous integers
    0..G'-1 so that downstream code can assume dense coding.
    """
    import torch

    if data.groups is None or min_group_size <= 1:
        return data

    codes = data.groups.numpy()
    counts = np.bincount(codes, minlength=data.n_groups)
    small = np.where(counts < min_group_size)[0]

    if len(small) == 0:
        return data

    # Build cell mask: keep cells whose group is not small
    mask = ~np.isin(codes, small)
    n_removed_cells = int((~mask).sum())
    removed_names = (
        [data.group_names[i] for i in small]
        if data.group_names is not None
        else list(small.astype(str))
    )
    print(
        f"  Removed {len(small)} small group(s) (<{min_group_size} cells): "
        + ", ".join(f"{n}({counts[c]})" for n, c in zip(removed_names, small))
    )
    print(f"  Removed {n_removed_cells} cells in small groups")

    # Re-encode surviving codes contiguously
    surviving_codes = np.unique(codes[mask])  # sorted old codes that remain
    code_map = np.full(data.n_groups, -1, dtype=np.int64)
    for new_code, old_code in enumerate(surviving_codes):
        code_map[old_code] = new_code
    new_codes = code_map[codes[mask]]

    new_group_names = (
        [data.group_names[c] for c in surviving_codes]
        if data.group_names is not None
        else None
    )
    new_n_groups = len(surviving_codes)

    t_mask = torch.from_numpy(mask)
    idx = np.where(mask)[0]
    spot_names = (
        [data.spot_names[i] for i in idx] if data.spot_names is not None else None
    )
    return SpatialData(
        X=data.X[t_mask],
        Y=data.Y[t_mask],
        groups=torch.from_numpy(new_codes).long(),
        n_groups=new_n_groups,
        gene_names=data.gene_names,
        spot_names=spot_names,
        group_names=new_group_names,
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

    # Drop groups with fewer than min_group_size cells and re-encode codes
    # min_group_fraction (e.g. 0.01) takes precedence over min_group_size if set
    min_group_fraction = config.preprocessing.get("min_group_fraction")
    if min_group_fraction is not None:
        min_group_size = max(1, round(min_group_fraction * data.n_spots))
        print(f"  min_group_size = {min_group_size} ({min_group_fraction*100:.1f}% of {data.n_spots:,})")
    else:
        min_group_size = config.preprocessing.get("min_group_size", 10)
    data = _filter_small_groups(data, min_group_size)
    print(f"  After small-group filter: {data.n_spots} spots, {data.n_groups} groups")

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
