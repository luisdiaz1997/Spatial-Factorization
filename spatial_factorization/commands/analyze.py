"""Analyze trained model (Stage 3).

Computes:
- Moran's I for spatial autocorrelation of each factor
- Reconstruction error (relative Frobenius norm)
- Poisson deviance (goodness-of-fit for count data)
- Factor values for each spot
- Group-specific loadings (loadings_group for each group g)
- Relative gene enrichment per factor per cell-type (group vs global loadings)

Outputs:
- metrics.json: All computed metrics
- factors.npy: Factor values (N, L)
- loadings.npy: Global loadings (D, L)
- loadings_group_{g}.npy: Group-specific loadings (D, L)
- moran_i.csv: Moran's I per factor with sort index
- gene_enrichment.json: Relative enrichment per factor per group
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from PNMF.transforms import get_factors, log_factors, transform_W

from ..config import Config
from ..datasets.base import load_preprocessed


def _load_model(model_dir: Path):
    """Load trained PNMF model from pickle."""
    with open(model_dir / "model.pkl", "rb") as f:
        return pickle.load(f)


def _compute_reconstruction_error(model, Y: np.ndarray) -> float:
    """Compute relative Frobenius norm reconstruction error.

    Args:
        model: Fitted PNMF model
        Y: Original data (N, D)

    Returns:
        Relative error: ||Y - Y_hat||_F / ||Y||_F
    """
    # Get factors and loadings
    F = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
    # components_ is (L, D), so we need (N, L) @ (L, D) = (N, D)
    Y_reconstructed = F @ model.components_  # (N, L) @ (L, D) = (N, D)

    # Relative Frobenius norm
    error = np.linalg.norm(Y - Y_reconstructed, 'fro') / np.linalg.norm(Y, 'fro')
    return float(error)


def _compute_poisson_deviance(model, Y: np.ndarray) -> dict:
    """Compute Poisson deviance (KL divergence) for goodness-of-fit.

    D(V_true || V_reconstruct) = sum_ij (V_ij * log(V_ij / mu_ij) - V_ij + mu_ij)

    For V_ij=0, the V*log(V/mu) term is 0, leaving just mu_ij.

    Args:
        model: Fitted PNMF model
        Y: Original data (N, D)

    Returns:
        Dictionary with deviance metrics:
        - total: Total Poisson deviance
        - mean: Mean deviance per observation
        - mean_per_gene: Mean deviance per gene (averaged over spots)
    """
    # Get reconstruction
    factors = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
    mu = factors @ model.components_  # (N, L) @ (L, D) = (N, D)

    # Ensure mu > 0 for numerical stability
    mu = np.maximum(mu, 1e-10)

    # Compute deviance: sum(y * log(y/mu) - y + mu)
    # For y=0: term is 0 - 0 + mu = mu
    # For y>0: term is y*log(y/mu) - y + mu
    deviance = np.zeros_like(Y, dtype=np.float64)

    # Where y > 0
    mask = Y > 0
    deviance[mask] = (
        Y[mask] * np.log(Y[mask] / mu[mask]) - Y[mask] + mu[mask]
    )

    # Where y = 0
    deviance[~mask] = mu[~mask]

    total_deviance = 2*float(np.sum(deviance))
    n_obs = Y.size
    n_genes = Y.shape[1]

    return {
        "total": total_deviance,
        "mean": total_deviance / n_obs,
        "mean_per_gene": total_deviance / n_genes,
    }


def _compute_group_loadings(
    model, Y: np.ndarray, groups: np.ndarray, max_iter: int = 1000, verbose: bool = False
) -> dict:
    """Compute group-specific loadings using PNMF's transform_W.

    Given the fitted model with factors F = mu (log-space), for each group g:
    - Extract Y_group = Y[groups == g]  (group_size, D)
    - Extract factors_group = mu[:, groups == g].T  (group_size, L) - log-space factors
    - Use transform_W to learn loadings_group such that Y_group ≈ exp(factors_group) @ loadings_group.T

    Args:
        model: Fitted PNMF model
        Y: Count matrix (N, D)
        groups: Group labels (N,)
        max_iter: Maximum iterations for transform_W optimization
        verbose: Whether to show progress

    Returns:
        Dictionary with:
        - loadings: dict mapping group_id -> loadings_group array (D, L)
        - reconstruction_error: dict mapping group_id -> relative error
    """
    # Get exp-space latent factors (N, L)
    F = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
    L = F.shape[1]

    # Get global loadings for fallback
    loadings_global = model.components_.T  # (D, L)

    unique_groups = np.unique(groups)
    loadings = {}
    recon_errors = {}

    for g in unique_groups:
        mask = groups == g
        Y_group = Y[mask]  # (group_size, D)
        F_group = F[mask]  # (group_size, L) - exp-space factors
        group_size = Y_group.shape[0]

        if group_size < L:
            # Not enough samples to estimate loadings
            loadings[int(g)] = loadings_global.copy()
            recon_errors[int(g)] = float("nan")
            continue

        # Use PNMF's transform_W to learn group-specific loadings
        loadings_group = transform_W(
            Y_group, F_group,
            max_iter=max_iter,
            verbose=verbose,
        )
        loadings[int(g)] = loadings_group

        # Compute reconstruction error for this group
        Y_group_hat = F_group @ loadings_group.T  # (group_size, L) @ (L, D) = (group_size, D)
        error = np.linalg.norm(Y_group - Y_group_hat, "fro") / np.linalg.norm(Y_group, "fro")
        recon_errors[int(g)] = float(error)

    return {"loadings": loadings, "reconstruction_error": recon_errors}


def _normalize_loadings(loadings: np.ndarray) -> np.ndarray:
    """Normalize loadings per factor to sum to 1 (like softmax over genes).

    This converts loadings to relative proportions, allowing comparison
    between global and group-specific loadings that have different scales.

    Args:
        loadings: (D, L) loadings matrix

    Returns:
        Normalized loadings where each column sums to 1
    """
    eps = 1e-10
    loadings_pos = np.maximum(loadings, eps)
    return loadings_pos / loadings_pos.sum(axis=0, keepdims=True)


def _compute_gene_enrichment(
    global_loadings: np.ndarray,
    group_loadings: dict,
    gene_names: list,
    group_names: list,
) -> dict:
    """Compute relative gene enrichment per factor by cell-type.

    For each factor and group, computes relative enrichment by:
    1. Normalizing loadings per factor (so they sum to 1 across genes)
    2. Computing log-fold change: log2(group_normalized / global_normalized)

    This normalization is critical because group-specific loadings are fit
    independently and have different scales than global loadings.

    Interpretation:
    - Positive LFC → gene has higher relative weight in this cell type
    - Negative LFC → gene has lower relative weight in this cell type

    Args:
        global_loadings: (D, L) global loadings
        group_loadings: dict mapping group_id -> (D, L) loadings
        gene_names: List of gene names
        group_names: List of group names

    Returns:
        Dictionary with enrichment results
    """
    D, L = global_loadings.shape
    gene_names = np.array(gene_names)

    # Normalize global loadings per factor
    global_norm = _normalize_loadings(global_loadings)

    eps = 1e-10

    results = {
        "n_factors": L,
        "n_groups": len(group_loadings),
        "group_names": group_names,
        "factors": {}
    }

    for factor_idx in range(L):
        global_loading = global_norm[:, factor_idx]

        factor_results = {
            "groups": {}
        }

        for group_id, group_loading_matrix in group_loadings.items():
            # Normalize group loadings per factor
            group_norm = _normalize_loadings(group_loading_matrix)
            group_loading = group_norm[:, factor_idx]

            # Log-fold change on normalized loadings (relative enrichment)
            lfc = np.log2((group_loading + eps) / (global_loading + eps))

            # Get top enriched (positive LFC) and depleted (negative LFC) genes
            top_enriched_idx = np.argsort(lfc)[-10:][::-1]
            top_depleted_idx = np.argsort(lfc)[:10]

            group_name = group_names[group_id] if group_id < len(group_names) else f"Group {group_id}"

            factor_results["groups"][group_name] = {
                "mean_lfc": float(np.mean(lfc)),
                "std_lfc": float(np.std(lfc)),
                "top_enriched": [
                    {"gene": gene_names[idx], "lfc": float(lfc[idx])}
                    for idx in top_enriched_idx
                ],
                "top_depleted": [
                    {"gene": gene_names[idx], "lfc": float(lfc[idx])}
                    for idx in top_depleted_idx
                ],
            }

        results["factors"][f"factor_{factor_idx}"] = factor_results

    return results


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
        metrics.json             - All computed metrics
        factors.npy              - Factor values (N, L)
        loadings.npy             - Global loadings (D, L)
        loadings_group_{g}.npy   - Group-specific loadings (D, L)
        moran_i.csv              - Moran's I per factor
        gene_enrichment.json     - Relative enrichment per factor per group
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
    factors = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
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

    # Compute Poisson deviance
    print("Computing Poisson deviance...")
    poisson_dev = _compute_poisson_deviance(model, data.Y.numpy())
    print(f"  Poisson deviance (mean): {poisson_dev['mean']:.4f}")

    # Compute group-specific loadings if groups exist
    group_loadings_result = None
    if data.groups is not None and data.n_groups > 1:
        print(f"\nComputing group-specific loadings for {data.n_groups} groups...")
        group_loadings_result = _compute_group_loadings(
            model, data.Y.numpy(), data.groups.numpy(), verbose=True
        )
        print(f"  Group reconstruction errors:")
        for g, err in group_loadings_result["reconstruction_error"].items():
            group_name = data.group_names[g] if data.group_names else f"Group {g}"
            print(f"    {group_name}: {err:.4f}")

    # Save factors
    np.save(model_dir / "factors.npy", factors)

    # Save global loadings
    np.save(model_dir / "loadings.npy", model.components_.T)  # (D, L)

    # Save group-specific loadings and compute gene enrichment
    gene_enrichment = None
    if group_loadings_result is not None:
        for g, loadings_group in group_loadings_result["loadings"].items():
            np.save(model_dir / f"loadings_group_{g}.npy", loadings_group)

        # Compute relative gene enrichment (group vs global loadings)
        print("\nComputing relative gene enrichment...")
        global_loadings = model.components_.T  # (D, L)
        gene_enrichment = _compute_gene_enrichment(
            global_loadings,
            group_loadings_result["loadings"],
            data.gene_names,
            data.group_names or [f"Group {g}" for g in range(data.n_groups)],
        )

        # Save enrichment results
        with open(model_dir / "gene_enrichment.json", "w") as f:
            json.dump(gene_enrichment, f, indent=2)
        print(f"  Saved gene enrichment for {gene_enrichment['n_factors']} factors x {gene_enrichment['n_groups']} groups")

    # Save Moran's I results
    moran_df = pd.DataFrame({
        "factor_idx": moran_idx,
        "moran_i": moran_values,
    })
    moran_df.to_csv(model_dir / "moran_i.csv", index=False)

    # Build metrics dict
    metrics = {
        "reconstruction_error": recon_error,
        "poisson_deviance": poisson_dev,
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

    # Add group-specific metrics if available
    if group_loadings_result is not None:
        metrics["group_loadings"] = {
            "n_groups": data.n_groups,
            "group_names": data.group_names,
            "reconstruction_errors": group_loadings_result["reconstruction_error"],
        }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAnalysis complete!")
    print(f"  Moran's I (mean):      {metrics['moran_i']['mean']:.4f}")
    print(f"  Moran's I (max):       {metrics['moran_i']['max']:.4f}")
    print(f"  Reconstruction error:  {recon_error:.4f}")
    print(f"  Poisson deviance:      {poisson_dev['mean']:.4f}")
    print(f"  Saved to: {model_dir}/")
