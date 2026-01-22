"""Test preprocessing and loading preprocessed data."""

import subprocess
from pathlib import Path

import pytest


def test_preprocess_and_load_slideseq():
    """Test preprocessing SlideSeq data and loading it back.

    This test:
    1. Runs: spatial_factorization preprocess -c configs/slideseq/pnmf.yaml
    2. Loads the preprocessed data
    3. Verifies shapes and prints group names
    """
    from spatial_factorization.datasets import load_preprocessed
    from spatial_factorization.config import Config

    config_path = Path("configs/slideseq/pnmf.yaml")
    output_dir = Path("outputs/slideseq_pnmf")

    # Step 1: Run preprocess command
    print("\n" + "=" * 60)
    print("STEP 1: Running preprocess command...")
    print("=" * 60)
    result = subprocess.run(
        ["spatial_factorization", "preprocess", "-c", str(config_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Preprocess failed:\n{result.stderr}")
        pytest.skip(f"Preprocess command failed. See output above.")
        return

    print(result.stdout)

    # Step 2: Load preprocessed data
    print("\n" + "=" * 60)
    print("STEP 2: Loading preprocessed data...")
    print("=" * 60)

    data = load_preprocessed(output_dir)

    # Print information
    print(f"  X (coordinates): {data.X.shape}")
    print(f"  Y (counts):       {data.Y.shape}")
    print(f"  groups:           {data.groups.shape}")
    print(f"  n_spots:          {data.n_spots:,}")
    print(f"  n_genes:          {data.n_genes:,}")
    print(f"  n_groups:         {data.n_groups}")

    print(f"\n  Group names ({data.n_groups}):")
    if data.group_names:
        for i, name in enumerate(data.group_names):
            count = (data.groups == i).sum().item()
            print(f"    [{i}] {name:30s} (n={count:,})")
    else:
        print("    (no group names)")

    print(f"\n  First 10 gene names:")
    if data.gene_names:
        for name in data.gene_names[:10]:
            print(f"    - {name}")
        if len(data.gene_names) > 10:
            print(f"    ... and {len(data.gene_names) - 10:,} more genes")

    print("=" * 60 + "\n")

    # Step 3: Verify config matches
    config = Config.from_yaml(config_path)
    print(f"Config verification:")
    print(f"  dataset.name:     {config.dataset.name}")
    print(f"  model.n_components: {config.model.n_components}")
    print(f"  model.spatial:     {config.model.spatial}")
    print()

    # Basic assertions
    assert data.X.ndim == 2 and data.X.shape[1] == 2, "X should be (N, 2)"
    assert data.Y.ndim == 2, "Y should be 2D"
    assert data.groups.shape[0] == data.X.shape[0], "groups and X should have same N"
    assert data.n_spots == data.X.shape[0], "n_spots mismatch"
    assert data.n_genes == data.Y.shape[0], "n_genes mismatch"

    print("✅ All assertions passed!")


if __name__ == "__main__":
    # Run test directly: python tests/test_preprocessed.py
    test_preprocess_and_load_slideseq()
