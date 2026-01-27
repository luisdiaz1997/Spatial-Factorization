"""Test preprocessing and loading preprocessed data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def load_and_verify_preprocessed(output_dir: Path, dataset_name: str):
    """Generic test to verify preprocessed data has all required fields.

    This test verifies:
    1. All required files exist (X.npy, Y.npy, C.npy, metadata.json)
    2. Shapes are consistent
    3. All metadata fields are populated
    4. Group names are provided
    5. Gene and spot names are provided

    Parameters
    ----------
    output_dir : Path
        Output directory containing preprocessed/ subdirectory.
    dataset_name : str
        Name of the dataset (for error messages).
    """
    prep_dir = output_dir / "preprocessed"

    # Check directory exists
    if not prep_dir.exists():
        pytest.skip(f"Preprocessed data not found for {dataset_name}: {prep_dir}")

    # Check all required files exist
    required_files = ["X.npy", "Y.npy", "C.npy", "metadata.json"]
    for fname in required_files:
        fpath = prep_dir / fname
        if not fpath.exists():
            pytest.fail(f"Missing required file: {fpath}")

    # Load arrays
    X = np.load(prep_dir / "X.npy")
    Y = np.load(prep_dir / "Y.npy")
    C = np.load(prep_dir / "C.npy")

    # Load metadata
    with open(prep_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Verify shapes: X is (N, 2), Y is (N, D), C is (N,)
    assert X.ndim == 2 and X.shape[1] == 2, f"{dataset_name}: X should be (N, 2)"
    assert Y.ndim == 2, f"{dataset_name}: Y should be 2D"
    assert C.ndim == 1, f"{dataset_name}: C should be 1D"

    N = X.shape[0]
    D = Y.shape[1]  # Y is (N, D) - spots x genes

    assert Y.shape[0] == N, f"{dataset_name}: Y shape mismatch (spots don't match X)"
    assert C.shape[0] == N, f"{dataset_name}: C shape mismatch (doesn't match X)"

    # Verify metadata fields
    required_meta_fields = ["n_spots", "n_genes", "n_groups", "gene_names",
                           "spot_names", "group_names", "dataset"]
    for field in required_meta_fields:
        if field not in metadata:
            pytest.fail(f"{dataset_name}: Missing metadata field: {field}")

    # Verify metadata matches array shapes
    assert metadata["n_spots"] == N, f"{dataset_name}: n_spots mismatch"
    assert metadata["n_genes"] == D, f"{dataset_name}: n_genes mismatch"

    # Verify gene_names and spot_names are populated
    gene_names = metadata["gene_names"]
    spot_names = metadata["spot_names"]
    group_names = metadata["group_names"]

    assert len(gene_names) == D, f"{dataset_name}: gene_names length mismatch"
    assert len(spot_names) == N, f"{dataset_name}: spot_names length mismatch"
    assert len(group_names) == metadata["n_groups"], f"{dataset_name}: group_names length mismatch"

    # Verify names are not empty
    assert all(name.strip() for name in gene_names[:10]), f"{dataset_name}: gene_names contain empty strings"
    assert all(name.strip() for name in spot_names[:10]), f"{dataset_name}: spot_names contain empty strings"
    assert all(name.strip() for name in group_names), f"{dataset_name}: group_names contain empty strings"

    # Verify group codes are valid
    n_groups = metadata["n_groups"]
    assert C.min() >= 0, f"{dataset_name}: Group codes contain negative values"
    assert C.max() < n_groups, f"{dataset_name}: Group codes exceed n_groups"

    # Print summary
    print(f"\n{dataset_name} verification:")
    print(f"  Spots (N):  {N:,}")
    print(f"  Genes (D):  {D:,}")
    print(f"  Groups:     {n_groups}")
    print(f"  Group names: {', '.join(group_names[:5])}{'...' if len(group_names) > 5 else ''}")
    print(f"[OK] All checks passed for {dataset_name}")


def test_slideseq_preprocessed():
    """Test SlideSeq preprocessed data."""
    load_and_verify_preprocessed(
        Path("outputs/slideseq"),
        "SlideSeq"
    )


def test_load_preprocessed_function():
    """Test the load_preprocessed() utility function.

    Verifies that the SpatialData object returned by load_preprocessed()
    has all expected fields with correct shapes.
    """
    from spatial_factorization.datasets import load_preprocessed

    output_dir = Path("outputs/slideseq")
    prep_dir = output_dir / "preprocessed"

    if not prep_dir.exists():
        pytest.skip(f"Preprocessed data not found: {prep_dir}")

    # Load using the utility function
    data = load_preprocessed(output_dir)

    # Verify SpatialData object
    assert isinstance(data.X, torch.Tensor), "X should be torch.Tensor"
    assert isinstance(data.Y, torch.Tensor), "Y should be torch.Tensor"
    assert isinstance(data.groups, torch.Tensor), "groups should be torch.Tensor"

    # Verify shapes
    assert data.X.shape[1] == 2, "X should be (N, 2)"
    assert data.Y.ndim == 2, "Y should be 2D"
    assert data.groups.shape[0] == data.X.shape[0], "groups and X should have same N"

    # Verify properties
    assert data.n_spots == data.X.shape[0], "n_spots mismatch"
    assert data.n_genes == data.Y.shape[1], "n_genes mismatch (Y is N x D)"

    # Verify names are populated
    assert data.gene_names is not None, "gene_names should be populated"
    assert data.spot_names is not None, "spot_names should be populated"
    assert data.group_names is not None, "group_names should be populated"

    assert len(data.gene_names) == data.n_genes, "gene_names length mismatch"
    assert len(data.spot_names) == data.n_spots, "spot_names length mismatch"
    assert len(data.group_names) == data.n_groups, "group_names length mismatch"

    print(f"\nload_preprocessed() verification:")
    print(f"  n_spots:     {data.n_spots:,}")
    print(f"  n_genes:     {data.n_genes:,}")
    print(f"  n_groups:    {data.n_groups}")
    print(f"[OK] SpatialData object is valid")


if __name__ == "__main__":
    # Run tests directly
    test_slideseq_preprocessed()
    test_load_preprocessed_function()
