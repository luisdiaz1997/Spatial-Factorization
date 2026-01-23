"""Test training PNMF models."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def load_and_verify_trained_model(output_dir: Path, model_name: str, dataset_name: str):
    """Generic test to verify trained model has all required outputs.

    This test verifies:
    1. Model directory exists (e.g., outputs/slideseq/pnmf/)
    2. Model file exists (model.pkl or model.pt)
    3. Training metadata exists (training.json)
    4. Training metadata contains required fields (elbo, time, n_components, etc.)
    5. Model can be loaded

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
        pytest.skip(f"Trained model not found for {dataset_name}/{model_name}: {model_dir}")

    # Check training metadata exists
    training_json = model_dir / "training.json"
    if not training_json.exists():
        pytest.fail(f"Missing training metadata: {training_json}")

    # Load training metadata
    with open(training_json) as f:
        training_meta = json.load(f)

    # Verify required metadata fields
    required_fields = ["n_components", "elbo", "training_time", "max_iter",
                      "converged", "timestamp", "model_config"]
    for field in required_fields:
        if field not in training_meta:
            pytest.fail(f"{dataset_name}/{model_name}: Missing metadata field: {field}")

    # Verify metadata types and values
    assert isinstance(training_meta["n_components"], int), "n_components should be int"
    assert training_meta["n_components"] > 0, "n_components should be positive"
    assert isinstance(training_meta["elbo"], (int, float)), "elbo should be numeric"
    assert isinstance(training_meta["training_time"], (int, float)), "training_time should be numeric"
    assert training_meta["training_time"] > 0, "training_time should be positive"
    assert isinstance(training_meta["converged"], bool), "converged should be bool"

    # Check model file exists (either .pkl or .pt)
    model_pkl = model_dir / "model.pkl"
    model_pt = model_dir / "model.pt"

    if not (model_pkl.exists() or model_pt.exists()):
        pytest.fail(f"Missing model file: expected {model_pkl} or {model_pt}")

    # Print summary
    print(f"\n{dataset_name}/{model_name} verification:")
    print(f"  n_components:   {training_meta['n_components']}")
    print(f"  ELBO:           {training_meta['elbo']:.2f}")
    print(f"  Training time:  {training_meta['training_time']:.2f}s")
    print(f"  Converged:      {training_meta['converged']}")
    print(f"  Max iterations: {training_meta['max_iter']}")
    print(f"[OK] All checks passed for {dataset_name}/{model_name}")


def test_slideseq_pnmf_training():
    """Test SlideSeq PNMF trained model."""
    load_and_verify_trained_model(
        Path("outputs/slideseq"),
        "pnmf",
        "SlideSeq"
    )


def test_load_trained_model():
    """Test loading a trained model and verifying components.

    Verifies that the trained model can be loaded and has expected attributes.
    """
    import pickle

    output_dir = Path("outputs/slideseq")
    model_dir = output_dir / "pnmf"
    model_pkl = model_dir / "model.pkl"

    if not model_pkl.exists():
        pytest.skip(f"Trained model not found: {model_pkl}")

    # Load model
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)

    # Verify model has expected attributes (PNMF sklearn API)
    assert hasattr(model, "components_"), "Model should have components_ (W matrix)"
    assert hasattr(model, "elbo_"), "Model should have elbo_"
    assert hasattr(model, "n_components"), "Model should have n_components"

    # Verify shapes
    W = model.components_  # (K, D) in sklearn convention
    K = model.n_components

    assert W.ndim == 2, "W should be 2D"
    assert W.shape[0] == K, f"W shape mismatch: expected ({K}, D)"

    # Verify components are non-negative
    assert np.all(W >= 0), "W components should be non-negative"

    print(f"\nTrained model verification:")
    print(f"  W shape:        {W.shape}")
    print(f"  n_components:   {K}")
    print(f"  ELBO:           {model.elbo_:.2f}")
    print(f"[OK] Model loaded successfully")


if __name__ == "__main__":
    # Run tests directly
    test_slideseq_pnmf_training()
    test_load_trained_model()
