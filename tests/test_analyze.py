"""Test analyze command (Stage 3)."""

import json
from pathlib import Path

import numpy as np
import pytest


def load_and_verify_analysis(output_dir: Path, model_name: str, dataset_name: str):
    """Generic test to verify analysis outputs.

    This test verifies:
    1. Analysis outputs exist (metrics.json, factors.npy, moran_i.csv)
    2. Metrics contain required fields
    3. Factors array has correct shape
    4. Moran's I values are valid

    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset (e.g., outputs/slideseq).
    model_name : str
        Name of the model (e.g., "pnmf", "lcgp").
    dataset_name : str
        Name of the dataset (for error messages).
    """
    model_dir = output_dir / model_name

    # Check model directory exists
    if not model_dir.exists():
        pytest.skip(f"Model directory not found for {dataset_name}/{model_name}: {model_dir}")

    # Check required analysis outputs exist
    metrics_json = model_dir / "metrics.json"
    factors_npy = model_dir / "factors.npy"
    moran_csv = model_dir / "moran_i.csv"

    if not metrics_json.exists():
        pytest.skip(f"Analysis not run yet: missing {metrics_json}")

    assert factors_npy.exists(), f"Missing factors.npy: {factors_npy}"
    assert moran_csv.exists(), f"Missing moran_i.csv: {moran_csv}"

    # Load metrics
    with open(metrics_json) as f:
        metrics = json.load(f)

    # Verify required fields
    required_fields = ["reconstruction_error", "moran_i", "n_factors", "n_spots", "timestamp"]
    for field in required_fields:
        assert field in metrics, f"Missing metrics field: {field}"

    # Verify moran_i structure
    moran = metrics["moran_i"]
    assert "values" in moran, "moran_i should have 'values'"
    assert "sorted_indices" in moran, "moran_i should have 'sorted_indices'"
    assert "mean" in moran, "moran_i should have 'mean'"
    assert "max" in moran, "moran_i should have 'max'"

    # Verify Moran's I values are valid (between -1 and 1)
    moran_values = np.array(moran["values"])
    assert np.all(moran_values >= -1) and np.all(moran_values <= 1), \
        "Moran's I values should be in [-1, 1]"

    # Load and verify factors
    factors = np.load(factors_npy)
    assert factors.ndim == 2, "Factors should be 2D"
    assert factors.shape[0] == metrics["n_spots"], "Factors N should match n_spots"
    assert factors.shape[1] == metrics["n_factors"], "Factors L should match n_factors"
    assert np.all(factors >= 0), "Factors should be non-negative (exp of latent)"

    # Print summary
    print(f"\n{dataset_name}/{model_name} analysis verification:")
    print(f"  n_spots:             {metrics['n_spots']}")
    print(f"  n_factors:           {metrics['n_factors']}")
    print(f"  Reconstruction err:  {metrics['reconstruction_error']:.4f}")
    print(f"  Moran's I (mean):    {moran['mean']:.4f}")
    print(f"  Moran's I (max):     {moran['max']:.4f}")
    print(f"  Moran's I (min):     {moran['min']:.4f}")
    print(f"[OK] All checks passed for {dataset_name}/{model_name} analysis")


def test_slideseq_pnmf_analysis():
    """Test SlideSeq PNMF analysis outputs."""
    load_and_verify_analysis(
        Path("outputs/slideseq"),
        "pnmf",
        "SlideSeq"
    )


def test_factors_shape_consistency():
    """Test that factors shape matches training metadata."""
    output_dir = Path("outputs/slideseq")
    model_dir = output_dir / "pnmf"

    if not model_dir.exists():
        pytest.skip("Model directory not found")

    training_json = model_dir / "training.json"
    factors_npy = model_dir / "factors.npy"

    if not training_json.exists() or not factors_npy.exists():
        pytest.skip("Training or analysis outputs not found")

    with open(training_json) as f:
        training = json.load(f)

    factors = np.load(factors_npy)

    # Verify shapes match
    assert factors.shape[1] == training["n_components"], \
        f"Factor dimensions should match n_components: {factors.shape[1]} != {training['n_components']}"
    assert factors.shape[0] == training["data_info"]["n_spots"], \
        f"Factor observations should match n_spots: {factors.shape[0]} != {training['data_info']['n_spots']}"

    print(f"\nShape consistency check:")
    print(f"  Factors: {factors.shape} (N, L)")
    print(f"  Training: n_spots={training['data_info']['n_spots']}, n_components={training['n_components']}")
    print(f"[OK] Shapes are consistent")


if __name__ == "__main__":
    # Run tests directly
    test_slideseq_pnmf_analysis()
    test_factors_shape_consistency()
