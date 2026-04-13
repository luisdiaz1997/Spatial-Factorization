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
import torch

from PNMF.transforms import get_factors, log_factors, transform_W, factor_uncertainty

from ..config import Config
from ..datasets.base import load_preprocessed


def _load_model(model_dir: Path, neighbors_override: str | None = None):
    """Load trained PNMF model from pickle or torch state dict.

    For spatial models, pickle fails due to MGGPWrapper, so we reconstruct
    the model architecture directly from GPzoo and load the state dict.

    When CUDA is not available, we prioritize .pth files which support map_location.
    """
    import torch

    pkl_path = model_dir / "model.pkl"
    pth_path = model_dir / "model.pth"

    # If CUDA is not available, try .pth first (it uses map_location='cpu')
    if not torch.cuda.is_available() and pth_path.exists():
        # Fall through to .pth loading below
        pass
    elif pkl_path.exists():
        # Try pickle first (works for non-spatial models on matching device)
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except (AttributeError, TypeError, EOFError, pickle.PicklingError, RuntimeError):
            # Pickle failed (likely spatial model or CUDA tensor on CPU-only), load from .pth
            pass

    # Load from torch state dict and reconstruct
    from PNMF import PNMF
    from PNMF.models import PoissonFactorization

    state = torch.load(model_dir / "model.pth", map_location="cpu", weights_only=False)
    hyperparams = state["hyperparameters"]

    is_spatial = hyperparams.get("spatial", False)
    n_components = hyperparams["n_components"]
    n_features = state["components"].shape[1]
    mode = hyperparams.get("mode", "expanded")
    loadings_mode = hyperparams.get("loadings_mode", "projected")

    # Create the PNMF wrapper (holds sklearn-like attributes)
    # Note: PNMF auto-selects prior based on spatial/multigroup/local flags
    model = PNMF(
        n_components=n_components,
        mode=mode,
        loadings_mode=loadings_mode,
        random_state=hyperparams.get("random_state", 0),
        spatial=is_spatial,
        multigroup=hyperparams.get("multigroup", False) if is_spatial else False,
    )

    if is_spatial:
        # Reconstruct spatial prior from saved state
        import torch.nn as nn
        from gpzoo.modules import CholeskyParameter

        prior_sd = state["prior_state_dict"]
        Z = prior_sd["Z"]                  # (M, 2)
        mu = prior_sd["mu"]                # (L, M)
        M = Z.shape[0]
        L = mu.shape[0]
        dim = Z.shape[1]

        # Detect multigroup vs standard SVGP from state dict
        is_multigroup = "groupsZ" in prior_sd
        if is_multigroup is None:
            # Fallback: check hyperparameters
            is_multigroup = hyperparams.get("multigroup", False)

        # Detect LCGP vs SVGP from state dict
        # LCGP has raw "Lu" parameter (L, M, R); SVGP has "Lu._raw" from CholeskyParameter
        is_lcgp = "Lu" in prior_sd and "Lu._raw" not in prior_sd

        if is_lcgp:
            # LCGP reconstruction path
            from gpzoo.gp import LCGP, MGGP_LCGP
            from gpzoo.kernels import batched_Matern32, batched_MGGP_Matern32

            K = hyperparams.get("K")
            if K is None:
                import warnings
                warnings.warn(
                    "Checkpoint is missing 'K' in hyperparameters (saved before K was persisted). "
                    "Defaulting to K=50. If state_dict loading fails, retrain to update checkpoint.",
                    UserWarning,
                )
                K = 50

            if is_multigroup:
                groupsZ = prior_sd["groupsZ"]      # (M,)
                n_groups = int(groupsZ.max().item()) + 1

                kernel = batched_MGGP_Matern32(
                    sigma=1.0, lengthscale=1.0,
                    group_diff_param=10.0, n_groups=n_groups,
                )
                gp = MGGP_LCGP(
                    kernel, dim=dim, M=M, n_groups=n_groups,
                    jitter=1e-5, K=K,
                )
            else:
                kernel = batched_Matern32(sigma=1.0, lengthscale=1.0)
                gp = LCGP(
                    kernel, dim=dim, M=M,
                    jitter=1e-5, K=K,
                )

            # Replace Lu with raw nn.Parameter matching saved shape (L, M, R)
            # R may differ from K when estimate_lcgp_rank clamps to a smaller value
            R = prior_sd["Lu"].shape[-1]
            del gp.Lu
            gp.Lu = nn.Parameter(torch.randn(L, M, R))
            gp.mu = nn.Parameter(torch.randn(L, M))

            # Load the trained prior state
            gp.load_state_dict(prior_sd)
            model._prior = gp

            # Set KNN indices (needed for LCGP forward pass + KL divergence)
            from gpzoo.knn_utilities import calculate_knn
            neighbors_strategy = neighbors_override or hyperparams.get("neighbors", "knn")
            raw = calculate_knn(
                gp, Z, strategy=neighbors_strategy,
                multigroup=is_multigroup,
                groupsX=groupsZ if is_multigroup else None,
                groupsZ=groupsZ if is_multigroup else None,
            )  # (N, K+1)
            gp.knn_idx = raw[:, :-1]   # self-inclusive — inference (forward pass)
            gp.knn_idz = raw[:, 1:]    # self-exclusive — KL divergence

            # Mark model as local and propagate neighbors strategy
            model.local = True
            model.neighbors = neighbors_strategy
            model.multigroup = is_multigroup
            # Needed by transforms._get_spatial_qF when calling probabilistic KNN
            model._groups = groupsZ if is_multigroup else None

        else:
            # SVGP/MGGP_SVGP reconstruction path
            if is_multigroup:
                from gpzoo.gp import MGGP_SVGP
                from gpzoo.kernels import batched_MGGP_Matern32

                groupsZ = prior_sd["groupsZ"]      # (M,)
                n_groups = int(groupsZ.max().item()) + 1

                kernel = batched_MGGP_Matern32(
                    sigma=1.0, lengthscale=1.0,
                    group_diff_param=10.0, n_groups=n_groups,
                )
                gp = MGGP_SVGP(
                    kernel, dim=dim, M=M, n_groups=n_groups,
                    jitter=1e-5, cholesky_mode="exp", diagonal_only=False,
                )
            else:
                from gpzoo.gp import SVGP
                from gpzoo.kernels import batched_Matern32

                kernel = batched_Matern32(
                    sigma=1.0, lengthscale=1.0,
                )
                gp = SVGP(
                    kernel, dim=dim, M=M,
                    jitter=1e-5, cholesky_mode="exp", diagonal_only=False,
                )

            # Replace mu and Lu with correct batched shapes (L, M) and (L, M, M)
            del gp.Lu
            gp.Lu = CholeskyParameter((L, M), mode="exp", diagonal_only=False)
            gp.mu = nn.Parameter(torch.randn(L, M) * 0.01)

            # Load the trained prior state
            gp.load_state_dict(prior_sd)
            model._prior = gp

        # Reconstruct PoissonFactorization with a dummy y (only shape matters)
        model_sd = state["model_state_dict"]
        D = n_features
        dummy_y = torch.empty(D, 1)
        model._model = PoissonFactorization(
            prior=gp, y=dummy_y, L=L,
            loadings_mode=loadings_mode, mode=mode,
        )
        model._model.load_state_dict(model_sd)
    else:
        # Non-spatial: reconstruct with GaussianPrior
        from PNMF.priors import GaussianPrior

        prior_sd = state["prior_state_dict"]
        model_sd = state["model_state_dict"]

        # Infer shapes from state dict
        # GaussianPrior.mean is (L, N)
        L, N = prior_sd["mean"].shape
        D = n_features  # Already computed from state["components"] above

        dummy_y = torch.empty(D, N)
        prior = GaussianPrior(y=dummy_y, L=L)
        prior.load_state_dict(prior_sd)
        model._prior = prior

        model._model = PoissonFactorization(
            prior=prior, y=dummy_y, L=L,
            loadings_mode=loadings_mode, mode=mode,
        )
        model._model.load_state_dict(model_sd)

    # Set sklearn-like attributes
    model.components_ = state["components"]
    model.n_components_ = n_components
    model.n_features_in_ = n_features

    return model


def _print_model_summary(model):
    """Print summary of learned model parameters."""
    print(f"  Components (L): {model.n_components_}")
    print(f"  Features (D):   {model.n_features_in_}")

    # Check if spatial (has a GP prior with kernel)
    is_spatial = hasattr(model, '_prior') and hasattr(model._prior, 'kernel')

    if is_spatial:
        kernel = model._prior.kernel
        sigma = kernel.sigma.data
        lengthscale = kernel.lengthscale.data

        # Handle scalar or batched kernel params
        if sigma.dim() == 0:
            print(f"  Kernel sigma:       {sigma.item():.4f}")
        else:
            print(f"  Kernel sigma:       {sigma.cpu().numpy()}")

        if lengthscale.dim() == 0:
            print(f"  Kernel lengthscale: {lengthscale.item():.4f}")
        else:
            print(f"  Kernel lengthscale: {lengthscale.cpu().numpy()}")

        # Group diff parameter (MGGP kernel)
        if hasattr(kernel, 'group_diff_param'):
            gdp = kernel.group_diff_param.data
            if gdp.dim() == 0:
                print(f"  Group diff param:   {gdp.item():.4f}")
            else:
                print(f"  Group diff param:   {gdp.cpu().numpy()}")

        # Inducing points and groups
        sd = model._prior.state_dict()
        if "Z" in sd:
            print(f"  Inducing points (M): {sd['Z'].shape[0]}")
        if "groupsZ" in sd:
            n_groups = int(sd["groupsZ"].max().item()) + 1
            print(f"  Groups:             {n_groups}")


def _get_factors_batched(model, coords: np.ndarray, groups, batch_size: int = 10000):
    """Get factors and scales from a spatial model in batches to avoid OOM.

    For SVGP, each batch avoids materialising the full (L, N, M) kernel matrix.
    For LCGP, each batch avoids the full (L, N, K, K) Su_knn tensor.

    KNN for LCGP is computed against the full stored gp.Z (all training points),
    so batching X is semantically correct — only the memory footprint shrinks.

    Returns:
        factors: (N, L) exp-space factor means
        scales:  (N, L) factor standard deviations
    """
    from PNMF.transforms import _get_spatial_qF

    N = coords.shape[0]
    all_means: list = []
    all_scales: list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        coords_b = coords[start:end]
        groups_b = groups[start:end] if groups is not None else None
        qF = _get_spatial_qF(model, coordinates=coords_b, groups=groups_b)
        all_means.append(torch.exp(qF.mean.detach().cpu()).T)   # (chunk, L)
        all_scales.append(qF.scale.detach().cpu().T)             # (chunk, L)

    factors = torch.cat(all_means, dim=0).numpy()   # (N, L)
    scales  = torch.cat(all_scales, dim=0).numpy()  # (N, L)
    return factors, scales


def _compute_reconstruction_error(model, Y: np.ndarray, F: np.ndarray = None,
                                   coordinates=None, groups=None) -> float:
    """Compute relative Frobenius norm reconstruction error.

    Args:
        model: Fitted PNMF model
        Y: Original data (N, D)
        F: Pre-computed (N, L) exp-space factors. If None, computed from model.
        coordinates: Spatial coordinates (used only when F is None)
        groups: Group labels (used only when F is None)

    Returns:
        Relative error: ||Y - Y_hat||_F / ||Y||_F
    """
    if F is None:
        if coordinates is not None:
            F = get_factors(model, use_mgf=False, coordinates=coordinates, groups=groups)
        else:
            F = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
    # components_ is (L, D), so we need (N, L) @ (L, D) = (N, D)
    Y_reconstructed = F @ model.components_  # (N, L) @ (L, D) = (N, D)

    # Relative Frobenius norm
    error = np.linalg.norm(Y - Y_reconstructed, 'fro') / np.linalg.norm(Y, 'fro')
    return float(error)


def _compute_poisson_deviance(model, Y: np.ndarray, F: np.ndarray = None,
                               coordinates=None, groups=None) -> dict:
    """Compute Poisson deviance (KL divergence) for goodness-of-fit.

    D(V_true || V_reconstruct) = sum_ij (V_ij * log(V_ij / mu_ij) - V_ij + mu_ij)

    For V_ij=0, the V*log(V/mu) term is 0, leaving just mu_ij.

    Args:
        model: Fitted PNMF model
        Y: Original data (N, D)
        F: Pre-computed (N, L) exp-space factors. If None, computed from model.
        coordinates: Spatial coordinates (used only when F is None)
        groups: Group labels (used only when F is None)

    Returns:
        Dictionary with deviance metrics:
        - total: Total Poisson deviance
        - mean: Mean deviance per observation
        - mean_per_gene: Mean deviance per gene (averaged over spots)
    """
    # Get reconstruction
    if F is None:
        if coordinates is not None:
            factors = get_factors(model, use_mgf=False, coordinates=coordinates, groups=groups)
        else:
            factors = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
    else:
        factors = F
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
    model, Y: np.ndarray, groups: np.ndarray, F: np.ndarray = None,
    coordinates=None, gp_groups=None,
    max_iter: int = 1000, verbose: bool = False,
) -> dict:
    """Compute group-specific loadings using PNMF's transform_W.

    Given the fitted model with factors F = mu (log-space), for each group g:
    - Extract Y_group = Y[groups == g]  (group_size, D)
    - Extract factors_group = mu[:, groups == g].T  (group_size, L) - log-space factors
    - Use transform_W to learn loadings_group such that Y_group ≈ exp(factors_group) @ loadings_group.T

    Args:
        model: Fitted PNMF model
        Y: Count matrix (N, D)
        groups: Group labels (N,) for subsetting data by cell-type
        F: Pre-computed (N, L) exp-space factors. If None, computed from model.
        coordinates: Spatial coordinates (used only when F is None)
        gp_groups: Group labels for GP conditioning (only for multigroup models).
            Pass None for non-multigroup spatial models (SVGP without groups).
        max_iter: Maximum iterations for transform_W optimization
        verbose: Whether to show progress

    Returns:
        Dictionary with:
        - loadings: dict mapping group_id -> loadings_group array (D, L)
        - reconstruction_error: dict mapping group_id -> relative error
    """
    # Get exp-space latent factors (N, L)
    if F is None:
        if coordinates is not None:
            F = get_factors(model, use_mgf=False, coordinates=coordinates, groups=gp_groups)
        else:
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
    """Normalize loadings per gene across factors (row normalization).

    For each gene d, divides its loading vector by the sum across all factors:
        loadings_norm[d, l] = loadings[d, l] / Σ_{l'} loadings[d, l']

    This captures each gene's *relative allocation* across factors, making
    enrichment comparisons (Wc_norm / W_norm) scale-invariant per gene.

    Args:
        loadings: (D, L) loadings matrix

    Returns:
        Normalized loadings where each row (gene) sums to 1
    """
    eps = 1e-10
    loadings_pos = np.maximum(loadings, eps)
    return loadings_pos / loadings_pos.sum(axis=1, keepdims=True)


def _compute_gene_enrichment(
    global_loadings: np.ndarray,
    group_loadings: dict,
    gene_names: list,
    group_names: list,
) -> dict:
    """Compute relative gene enrichment per factor by cell-type.

    For each factor and group, computes relative enrichment by:
    1. Normalizing loadings per gene across factors (each gene's row sums to 1):
           Wc_norm[d, l] = Wc[d, l] / Σ_{l'} Wc[d, l']
           W_norm[d, l]  = W[d, l]  / Σ_{l'} W[d, l']
    2. Computing log-fold change: log2(Wc_norm / W_norm)

    Row normalization makes each gene scale-invariant: we compare how a gene
    *allocates* its loading across factors in group c vs globally, ignoring
    the overall magnitude. This is robust to W being sparser than Wc.

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


def _get_groupwise_factors_batched(
    model, coords: np.ndarray, n_groups: int,
    sort_order: np.ndarray, batch_size: int = 10000,
) -> dict:
    """Compute conditional posterior factors for MGGP models.

    For each group g in {0, ..., n_groups-1}, runs the GP forward pass with
    all cells forced to group g (groupsX_g = full(g)), giving the posterior
    factor map under the kernel conditioned on that cell type.

    Args:
        model:      Fitted PNMF model with MGGP prior
        coords:     (N, 2) spatial coordinates
        n_groups:   Number of groups G
        sort_order: Moran's I sort indices (applied to factor columns)
        batch_size: Batch size for GP forward pass

    Returns:
        dict {g: (N, L) float32 array} — exp-space posterior means, Moran's I sorted
    """
    from PNMF.transforms import _get_spatial_qF

    N = coords.shape[0]
    result = {}

    for g in range(n_groups):
        print(f"  Group {g + 1}/{n_groups}...", end="\r", flush=True)
        all_means = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            coords_b = coords[start:end]
            groups_b = np.full(end - start, g, dtype=np.int64)
            qF = _get_spatial_qF(model, coordinates=coords_b, groups=groups_b)
            all_means.append(torch.exp(qF.mean.detach().cpu()).T)  # (chunk, L)
        factors_g = torch.cat(all_means, dim=0).numpy()  # (N, L)
        factors_g = factors_g[:, sort_order]             # apply Moran's I sort
        result[g] = factors_g.astype(np.float32)

    print()  # newline after \r progress
    return result


def _compute_chunk_posterior_params(knn_idx, mu, Lu_raw, Z, groupsZ, kernel, jitter, add_jitter_fn):
    """Compute Kzz cholesky and transformed mu for a chunk of KNN indices.

    Returns: (L_kzz, Z_nbr, gZ_nbr, mu_t) tensors for posterior computation.
    """
    mu_knn = mu[:, knn_idx]                           # (L, c, K_post)
    Lu_knn = Lu_raw[:, knn_idx]                       # (L, c, K_post, R)
    Su_knn = Lu_knn @ Lu_knn.transpose(-2, -1)        # (L, c, K_post, K_post)
    add_jitter_fn(Su_knn, jitter)
    L_su = torch.linalg.cholesky(Su_knn)              # (L, c, K_post, K_post)
    del Lu_knn, Su_knn

    Z_nbr = Z[knn_idx]                                # (c, K_post, 2)
    gZ_nbr = groupsZ[knn_idx]                         # (c, K_post)

    Kzz = torch.vmap(lambda z, gz: kernel(z, z, gz, gz))(Z_nbr, gZ_nbr).contiguous()
    add_jitter_fn(Kzz, jitter)
    L_kzz = torch.linalg.cholesky(Kzz)               # (c, K_post, K_post)
    del Kzz

    stacked = torch.cat([mu_knn.unsqueeze(-1), L_su], dim=-1)
    X_sol = torch.linalg.solve_triangular(L_kzz, stacked, upper=False)
    mu_t = X_sol[..., 0]     # (L, c, K_post)
    del stacked, X_sol, L_su, mu_knn

    return L_kzz, Z_nbr, gZ_nbr, mu_t


def _compute_group_posterior(g, L_kzz, Z_nbr, gZ_nbr, mu_t, X_chunk, kernel, device, sort_order):
    """Compute posterior factors for a single group given precomputed Kzz params."""
    c = X_chunk.shape[0]
    gX_g = torch.full((c, 1), g, dtype=torch.long, device=device)

    Kzx_g = torch.vmap(
        lambda z, x, gz, gx: kernel(z, x, gz, gx)
    )(Z_nbr, X_chunk, gZ_nbr, gX_g)              # (c, K_post, 1)

    Wt_g = torch.linalg.solve_triangular(L_kzz, Kzx_g, upper=False)
    W_g = Wt_g.transpose(-2, -1)                  # (c, 1, K_post)

    mean_g = (W_g @ mu_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (L, c)
    factors_g = torch.exp(mean_g).T.cpu().numpy()  # (c, L)
    return factors_g[:, sort_order]


def _get_groupwise_factors_expanded_K(
    model, coords, groups, n_groups: int,
    sort_order: np.ndarray,
    K_post: int = 2000,
    mem_gb: float = 8.0,
    knn_groups_mode: str = "original",
) -> dict:
    """Expanded-K posterior for MGGP_LCGP groupwise factors.

    Uses K_post >> K_train neighbors for a smoother posterior. Only valid for
    LCGP/MGGP_LCGP models. MGGP_SVGP should use _get_groupwise_factors_batched.

    Args:
        model:      Fitted PNMF model with MGGP_LCGP prior
        coords:     (N, 2) spatial coordinates (numpy or tensor)
        groups:     (N,) actual group codes for all points (numpy int64)
        n_groups:   Number of groups G
        sort_order: Moran's I sort indices (applied to factor columns)
        K_post:     Number of posterior neighbors (>> K_train)
        mem_gb:     GPU memory budget in GB (controls chunk size)
        knn_groups_mode: How to compute groupsX for probabilistic KNN:
            - "original": compute KNN once with actual groups [0,2,3,3,...] (default)
            - "derived": compute KNN per group with groupsX=[g,g,g,...] (old batched behavior)

    Returns:
        dict {g: (N, L) float32 array} — exp-space posterior means, Moran's I sorted
    """
    from gpzoo.utilities import add_jitter
    from gpzoo.knn_utilities import _faiss_knn, _probabilistic_knn

    gp = model._prior
    kernel = gp.kernel
    mu = gp.mu                    # (L, M)
    Lu_raw = gp.Lu                # (L, M, R)
    Z = gp.Z                      # (M, 2)
    groupsZ = gp.groupsZ          # (M,)

    strategy = getattr(model, 'neighbors', 'knn')
    L = mu.shape[0]
    N = coords.shape[0] if hasattr(coords, 'shape') else len(coords)
    device = mu.device

    # Adaptive chunk size: ~4 live (L, C, K_post, K_post) tensors
    C = max(1, int(mem_gb * 1e9 / (L * K_post ** 2 * 4 * 4)))

    # Convert coords to tensor
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords.astype(np.float32)).to(device)
    else:
        coords_t = coords.to(device)

    # Convert groups to CPU tensor for KNN computation
    if isinstance(groups, np.ndarray):
        groups_t = torch.from_numpy(groups.astype(np.int64))
    else:
        groups_t = groups.cpu()

    # Compute K_post neighbors — mode determines groupsX for probabilistic KNN
    # "original": shared KNN with actual groups [0,2,3,3,...] (default)
    # "derived": per-group KNN with groupsX=[g,g,g,...] (matches old _get_groupwise_factors_batched)
    use_per_group_knn = (knn_groups_mode == "derived" and strategy == "probabilistic")

    if use_per_group_knn:
        print(f"  Computing K_post={K_post} neighbors per group (strategy={strategy}, mode=derived)...", flush=True)
        knn_idx_per_group = {}
        for g in range(n_groups):
            groups_g = torch.full((N,), g, dtype=torch.long)
            knn_idx_per_group[g] = _probabilistic_knn(
                coords_t, Z, K_post, kernel,
                multigroup=True, groupsX=groups_g, groupsZ=groupsZ.cpu(),
            )[:, :-1]
    else:
        print(f"  Computing K_post={K_post} neighbors (strategy={strategy}, mode={knn_groups_mode})...", flush=True)
        if strategy == 'knn':
            knn_idx_all = _faiss_knn(coords_t, Z, K_post)[:, :-1]
        else:
            knn_idx_all = _probabilistic_knn(
                coords_t, Z, K_post, kernel,
                multigroup=True, groupsX=groups_t, groupsZ=groupsZ.cpu(),
            )[:, :-1]

    result = {g: np.empty((N, L), dtype=np.float32) for g in range(n_groups)}
    n_chunks = (N + C - 1) // C

    with torch.no_grad():
        for chunk_i, start in enumerate(range(0, N, C)):
            end = min(start + C, N)
            c = end - start
            print(f"  Chunk {chunk_i + 1}/{n_chunks} ({c} pts, K_post={K_post})...",
                  end="\r", flush=True)

            X_chunk = coords_t[start:end].unsqueeze(1)  # (c, 1, 2)

            if use_per_group_knn:
                # Per-group mode: compute Kzz/mu_t per group
                for g in range(n_groups):
                    print(f"  Chunk {chunk_i + 1}/{n_chunks}, Group {g + 1}/{n_groups}...",
                          end="\r", flush=True)
                    knn_idx = knn_idx_per_group[g][start:end].to(device)
                    L_kzz, Z_nbr, gZ_nbr, mu_t = _compute_chunk_posterior_params(
                        knn_idx, mu, Lu_raw, Z, groupsZ, kernel, gp.jitter, add_jitter
                    )
                    result[g][start:end] = _compute_group_posterior(
                        g, L_kzz, Z_nbr, gZ_nbr, mu_t, X_chunk, kernel, device, sort_order
                    )
                    del L_kzz, Z_nbr, gZ_nbr, mu_t
            else:
                # Shared KNN mode: compute Kzz/mu_t once, reuse for all groups
                knn_idx = knn_idx_all[start:end].to(device)
                L_kzz, Z_nbr, gZ_nbr, mu_t = _compute_chunk_posterior_params(
                    knn_idx, mu, Lu_raw, Z, groupsZ, kernel, gp.jitter, add_jitter
                )
                for g in range(n_groups):
                    result[g][start:end] = _compute_group_posterior(
                        g, L_kzz, Z_nbr, gZ_nbr, mu_t, X_chunk, kernel, device, sort_order
                    )
                del L_kzz, Z_nbr, gZ_nbr, mu_t

    print()  # newline after \r progress
    return result


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


def run(config_path: str, probabilistic: bool = False, posterior_k: int | None = None, posterior_mem_gb: float | None = None, knn_groups_mode: str = "original"):
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
    model_dir = output_dir / config.model_name

    # Load preprocessed data
    print(f"Loading preprocessed data from: {output_dir}/preprocessed/")
    data = load_preprocessed(output_dir)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Load trained model
    print(f"Loading trained model from: {model_dir}/")
    if probabilistic:
        print("[--probabilistic] Overriding KNN strategy to 'probabilistic' for this analyze run")
    model = _load_model(model_dir, neighbors_override="probabilistic" if probabilistic else None)
    _print_model_summary(model)

    # Extract factors (for spatial models, pass coordinates and optionally groups)
    spatial = config.spatial
    use_groups = config.groups
    print("Extracting factors...")
    coords = data.X.numpy()
    groups = data.groups.numpy() if (data.groups is not None and use_groups) else None

    # analyze_batch_size: chunk N to avoid OOM on large datasets (SVGP/LCGP)
    analyze_batch_size = config.training.get("analyze_batch_size", 10000)

    if spatial:
        # Batched GP forward pass — avoids OOM for large N
        factors, scales = _get_factors_batched(model, coords, groups, analyze_batch_size)
    else:
        factors = get_factors(model, use_mgf=False)  # (N, L) - exp(μ)
        scales = factor_uncertainty(model, return_variance=False)  # (N, L)
    print(f"  Factors shape: {factors.shape}")

    # Compute Moran's I for spatial autocorrelation
    print("Computing Moran's I for spatial autocorrelation...")
    moran_idx, moran_values = _compute_moran_i(factors, coords)
    print(f"  Top 3 Moran's I: {moran_values[:3]}")

    # Compute reconstruction error (use pre-computed factors to avoid second GP pass)
    print("Computing reconstruction error...")
    recon_error = _compute_reconstruction_error(model, data.Y.numpy(), F=factors)
    print(f"  Reconstruction error: {recon_error:.4f}")

    # Compute Poisson deviance (use pre-computed factors to avoid third GP pass)
    print("Computing Poisson deviance...")
    poisson_dev = _compute_poisson_deviance(model, data.Y.numpy(), F=factors)
    print(f"  Poisson deviance (mean): {poisson_dev['mean']:.4f}")

    # Compute group-specific loadings whenever data has groups
    # (works for all models: MGGP_SVGP, SVGP, and PNMF)
    group_loadings_result = None
    if data.groups is not None and data.n_groups > 1:
        print(f"\nComputing group-specific loadings for {data.n_groups} groups...")
        # Pass pre-computed factors to avoid a fourth GP forward pass
        group_loadings_result = _compute_group_loadings(
            model, data.Y.numpy(), data.groups.numpy(),
            F=factors, verbose=True,
        )
        print(f"  Group reconstruction errors:")
        for g, err in group_loadings_result["reconstruction_error"].items():
            group_name = data.group_names[g] if data.group_names else f"Group {g}"
            print(f"    {group_name}: {err:.4f}")

    # Reorder factors by Moran's I (descending) so factor 0 has highest spatial autocorrelation
    print("\nReordering factors by Moran's I (descending)...")
    sort_order = moran_idx
    factors = factors[:, sort_order]
    scales = scales[:, sort_order]
    model.components_ = model.components_[sort_order, :]
    if group_loadings_result is not None:
        for g in group_loadings_result["loadings"]:
            group_loadings_result["loadings"][g] = group_loadings_result["loadings"][g][:, sort_order]
    print(f"  Factor order (by Moran's I): {sort_order.tolist()}")

    # Reorder video frames if they exist (same sort_order as factors, avoids re-running Moran's I)
    video_frames_path = model_dir / "video_frames.npy"
    if video_frames_path.exists():
        print("  Reordering video frames to match factor order...")
        video_frames = np.load(video_frames_path)          # (n_frames, N, L)
        video_frames = video_frames[:, :, sort_order]
        np.save(video_frames_path, video_frames)
        np.save(model_dir / "video_moran_values.npy", moran_values)
        print(f"  Saved reordered video frames and Moran's I cache.")

    # Save factors
    np.save(model_dir / "factors.npy", factors)

    # Save factor uncertainty (scales) - already computed and reordered above
    np.save(model_dir / "scales.npy", scales)
    print(f"  Scales shape: {scales.shape}, mean: {scales.mean():.4f}")

    # Save global loadings
    np.save(model_dir / "loadings.npy", model.components_.T)  # (D, L)

    # Save inducing-point data for spatial models (used by figures stage)
    if spatial:
        import torch as _torch
        gp = model._prior
        sd = gp.state_dict()
        Z = sd["Z"]                  # (M, 2) inducing locations

        # Check if LCGP (raw nn.Parameter Lu) vs SVGP (CholeskyParameter Lu)
        # SVGP's CholeskyParameter has a _raw attribute; LCGP's plain Parameter does not
        is_lcgp = not hasattr(gp.Lu, '_raw')

        if is_lcgp:
            # LCGP: Save Lu as raw parameter (L, M, R)
            Lu_raw = gp.Lu.data[sort_order, :, :]  # (L, M, R) reordered by Moran's I
            np.save(model_dir / "Lu.npy", Lu_raw.detach().cpu().numpy())
            msg = f"  Saved LCGP Lu data: Lu {tuple(Lu_raw.shape)}, Z {tuple(Z.shape)}"
        else:
            # SVGP: Save full Cholesky factor
            Lu_constrained = gp.Lu.data[sort_order, :, :]  # (L, M, M) reordered by Moran's I
            _torch.save(Lu_constrained, model_dir / "Lu.pt")
            msg = f"  Saved inducing-point data: Lu {tuple(Lu_constrained.shape)}, Z {tuple(Z.shape)}"

        np.save(model_dir / "Z.npy", Z.detach().cpu().numpy())
        if "groupsZ" in sd:
            groupsZ = sd["groupsZ"]  # (M,) inducing point group assignments
            np.save(model_dir / "groupsZ.npy", groupsZ.detach().cpu().numpy())
            msg += f", groupsZ {tuple(groupsZ.shape)}"
        print(msg)

    # Compute and save groupwise conditional posterior (MGGP models only)
    if config.groups and spatial:
        print(f"\nComputing groupwise conditional posterior ({data.n_groups} groups)...")
        groupwise_dir = model_dir / "groupwise_factors"
        groupwise_dir.mkdir(exist_ok=True)

        # Use expanded-K posterior for LCGP models when posterior_K is configured
        _posterior_k = posterior_k or config.training.get("posterior_K")
        _is_lcgp = not hasattr(model._prior.Lu, '_raw')

        if _is_lcgp and _posterior_k is not None:
            _mem_gb = posterior_mem_gb or config.training.get("posterior_mem_gb", 8.0)
            print(f"  Using expanded-K posterior: K_post={_posterior_k}, mem_gb={_mem_gb}, knn_mode={knn_groups_mode}")
            groupwise = _get_groupwise_factors_expanded_K(
                model, coords, groups, data.n_groups, sort_order,
                K_post=_posterior_k, mem_gb=_mem_gb, knn_groups_mode=knn_groups_mode,
            )
        else:
            groupwise = _get_groupwise_factors_batched(
                model, coords, data.n_groups, sort_order, analyze_batch_size
            )
        for g, factors_g in groupwise.items():
            np.save(groupwise_dir / f"group_{g}.npy", factors_g)
        print(f"  Saved {data.n_groups} groupwise factor arrays to {groupwise_dir}/")

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

        # Compute PCA gene order per factor: for each factor, stack LFC across groups (D, G),
        # run PCA, order genes by PC1. Result shape: (L, D).
        print("\nComputing PCA gene ordering (per factor)...")
        from sklearn.decomposition import PCA
        D, L = model.components_.T.shape
        eps = 1e-10
        # Normalize global loadings per gene across factors
        global_raw = np.maximum(model.components_.T, eps)  # (D, L)
        global_norm = global_raw / global_raw.sum(axis=1, keepdims=True)
        # Precompute per-group normalized LFC arrays
        sorted_groups = sorted(group_loadings_result["loadings"])
        lfc_by_group = []
        for g in sorted_groups:
            grp_raw = np.maximum(group_loadings_result["loadings"][g], eps)  # (D, L)
            grp_norm = grp_raw / grp_raw.sum(axis=1, keepdims=True)
            lfc_by_group.append(np.log2(grp_norm / global_norm))  # (D, L)
        pca = PCA(n_components=1)
        pca_gene_order = np.zeros((L, D), dtype=int)
        for l in range(L):
            # Stack across groups for this factor: (D, G)
            factor_lfc = np.stack([lfc[:, l] for lfc in lfc_by_group], axis=1)
            pca.fit(factor_lfc)
            pc1_scores = pca.transform(factor_lfc)[:, 0]  # (D,)
            pca_gene_order[l] = np.argsort(pc1_scores)[::-1]
        np.save(model_dir / "pca_gene_order.npy", pca_gene_order)
        print(f"  Saved pca_gene_order.npy ({L} factors x {D} genes, per-factor PC1 ordering)")

        # Compute PCA gene order per cell type: for each group, use LFC (D, L),
        # run PCA, order genes by PC1. Result shape: (G, D).
        G = len(sorted_groups)
        pca_gene_order_by_celltype = np.zeros((G, D), dtype=int)
        for i, lfc in enumerate(lfc_by_group):
            # lfc is (D, L) — genes as samples, factors as features
            pca.fit(lfc)
            pc1_scores = pca.transform(lfc)[:, 0]  # (D,)
            pca_gene_order_by_celltype[i] = np.argsort(pc1_scores)[::-1]
        np.save(model_dir / "pca_gene_order_by_celltype.npy", pca_gene_order_by_celltype)
        print(f"  Saved pca_gene_order_by_celltype.npy ({G} cell types x {D} genes, per-celltype PC1 ordering)")

    # Save Moran's I results
    moran_df = pd.DataFrame({
        "factor_idx": np.arange(len(moran_values)),
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
