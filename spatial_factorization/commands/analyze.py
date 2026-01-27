"""Analyze trained model (Stage 3).

Computes:
- Moran's I for spatial autocorrelation of each factor
- Reconstruction error (relative Frobenius norm)
- Factor values for each spot

Outputs:
- metrics.json: All computed metrics
- factors.npy: Factor values (N, L)
- moran_i.csv: Moran's I per factor with sort index
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..datasets.base import load_preprocessed


def _load_model(model_dir: Path):
    """Load trained PNMF model from pickle."""
    with open(model_dir / "model.pkl", "rb") as f:
        return pickle.load(f)


def _extract_factors(model) -> np.ndarray:
    """Extract factor values from fitted PNMF model.

    Returns:
        factors: (N, L) array of factor values (exp of latent mean)
    """
    # PNMF stores qF prior with mean shape (L, N)
    # We want (N, L) for compatibility with dims_autocorr
    latent_mean = model._prior.mean.detach().cpu().numpy()  # (L, N)
    factors = np.exp(latent_mean).T  # (N, L)
    return factors


def _compute_reconstruction_error(model, Y: np.ndarray) -> float:
    """Compute relative Frobenius norm reconstruction error.

    Args:
        model: Fitted PNMF model
        Y: Original data (N, D)

    Returns:
        Relative error: ||Y - Y_hat||_F / ||Y||_F
    """
    # Get factors and loadings
    factors = _extract_factors(model)  # (N, L)
    W = model.components_.T  # (L, D) -> transpose to (D, L) then back? No, W is (L, D)
    # Actually components_ is (L, D), so we need (N, L) @ (L, D) = (N, D)
    Y_reconstructed = factors @ model.components_  # (N, L) @ (L, D) = (N, D)

    # Relative Frobenius norm
    error = np.linalg.norm(Y - Y_reconstructed, 'fro') / np.linalg.norm(Y, 'fro')
    return float(error)


def _compute_moran_i(factors: np.ndarray, coords: np.ndarray) -> tuple:
    """Compute Moran's I for each factor using squidpy.

    Args:
        factors: (N, L) array of factor values
        coords: (N, 2) spatial coordinates

    Returns:
        moran_idx: Indices sorted by Moran's I (descending)
        moran_values: Moran's I values for each factor (in sorted order)
    """
    try:
        from gpzoo.utilities import dims_autocorr
        return dims_autocorr(factors, coords, sort=True)
    except ImportError:
        # Fallback: compute directly with squidpy
        from anndata import AnnData
        from squidpy.gr import spatial_neighbors, spatial_autocorr

        ad = AnnData(X=factors, obsm={"spatial": coords})
        spatial_neighbors(ad)
        df = spatial_autocorr(ad, mode="moran", copy=True)

        idx = np.array([int(i) for i in df.index])
        return idx, df["I"].to_numpy()


def run(config_path: str):
    """Analyze a trained PNMF model.

    Output files (outputs/{dataset}/{model}/):
        metrics.json   - All computed metrics
        factors.npy    - Factor values (N, L)
        moran_i.csv    - Moran's I per factor
    """
    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)

    # Determine model directory
    spatial = config.model.get("spatial", False)
    prior = config.model.get("prior", "GaussianPrior")
    model_name = prior.lower() if spatial else "pnmf"
    model_dir = output_dir / model_name

    # Load preprocessed data
    print(f"Loading preprocessed data from: {output_dir}/preprocessed/")
    data = load_preprocessed(output_dir)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Load trained model
    print(f"Loading trained model from: {model_dir}/")
    model = _load_model(model_dir)

    # Extract factors
    print("Extracting factors...")
    factors = _extract_factors(model)
    print(f"  Factors shape: {factors.shape}")

    # Compute Moran's I for spatial autocorrelation
    print("Computing Moran's I for spatial autocorrelation...")
    coords = data.X.numpy()
    moran_idx, moran_values = _compute_moran_i(factors, coords)
    print(f"  Top 3 Moran's I: {moran_values[:3]}")

    # Compute reconstruction error
    print("Computing reconstruction error...")
    recon_error = _compute_reconstruction_error(model, data.Y.numpy())
    print(f"  Reconstruction error: {recon_error:.4f}")

    # Save factors
    np.save(model_dir / "factors.npy", factors)

    # Save Moran's I results
    moran_df = pd.DataFrame({
        "factor_idx": moran_idx,
        "moran_i": moran_values,
    })
    moran_df.to_csv(model_dir / "moran_i.csv", index=False)

    # Build metrics dict
    metrics = {
        "reconstruction_error": recon_error,
        "moran_i": {
            "values": moran_values.tolist(),
            "sorted_indices": moran_idx.tolist(),
            "mean": float(np.mean(moran_values)),
            "max": float(np.max(moran_values)),
            "min": float(np.min(moran_values)),
        },
        "n_factors": factors.shape[1],
        "n_spots": factors.shape[0],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAnalysis complete!")
    print(f"  Moran's I (mean): {metrics['moran_i']['mean']:.4f}")
    print(f"  Moran's I (max):  {metrics['moran_i']['max']:.4f}")
    print(f"  Reconstruction:   {recon_error:.4f}")
    print(f"  Saved to: {model_dir}/")
