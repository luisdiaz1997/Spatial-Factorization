"""Train PNMF model (Stage 2)."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PNMF import PNMF

from ..config import Config
from ..datasets.base import load_preprocessed


def _save_model(model, config: Config, model_dir: Path) -> None:
    """Save trained model to disk (both pickle and torch formats)."""
    # Pickle: Full model with sklearn API
    # Note: Spatial models with MGGP prior can't be pickled due to local class wrapper
    try:
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    except (AttributeError, TypeError) as e:
        print(f"  Warning: Could not pickle model ({e}), saving torch state dict only")

    # PyTorch state_dict: More portable, version-independent
    state = {
        "model_state_dict": model._model.state_dict(),
        "prior_state_dict": model._prior.state_dict(),
        "components": model.components_,
        "hyperparameters": {
            "n_components": model.n_components,
            "mode": config.model.get("mode", "expanded"),
            "loadings_mode": config.model.get("loadings_mode", "projected"),
            "random_state": config.seed,
        },
    }

    # Add spatial-specific info for spatial models
    if config.spatial:
        state["hyperparameters"]["spatial"] = True
        state["hyperparameters"]["prior"] = config.prior

    torch.save(state, model_dir / "model.pth")


def _save_elbo_history(elbo_history: list, model_dir: Path) -> None:
    """Save ELBO history (both CSV and numpy formats)."""
    df = pd.DataFrame({"iteration": range(len(elbo_history)), "elbo": elbo_history})
    df.to_csv(model_dir / "elbo_history.csv", index=False)
    np.save(model_dir / "elbo_history.npy", np.array(elbo_history))


def _get_training_metadata(model, config: Config, data, train_time: float) -> dict:
    """Build training metadata dict."""
    return {
        "n_components": model.n_components_,
        "elbo": float(model.elbo_),
        "training_time": train_time,
        "max_iter": config.training.get("max_iter", 10000),
        "converged": model.n_iter_ < config.training.get("max_iter", 10000),
        "n_iterations": model.n_iter_,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": config.model,
        "training_config": config.training,
        "data_info": {
            "n_spots": data.n_spots,
            "n_genes": data.n_genes,
            "n_groups": data.n_groups,
        },
    }


def run(config_path: str):
    """Train a PNMF model from config.

    Output files (outputs/{dataset}/{model}/):
        model.pkl          - Trained PNMF model (pickle)
        model.pth          - PyTorch state dict
        training.json      - Training metadata
        elbo_history.csv   - ELBO history
        config.yaml        - Copy of config used
    """
    config = Config.from_yaml(config_path)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load preprocessed data
    output_dir = Path(config.output_dir)
    print(f"Loading preprocessed data from: {output_dir}/preprocessed/")
    data = load_preprocessed(output_dir)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Build model (data.Y is already in N x D format)
    n_components = config.model.get("n_components", 10)
    mode = config.model.get("mode", "expanded")
    device = config.training.get("device", "cpu")
    print(f"Training PNMF with {n_components} components (mode={mode}, device={device})...")

    model = PNMF(random_state=config.seed, **config.to_pnmf_kwargs())

    # Train (spatial models require coordinates and groups)
    t0 = time.perf_counter()
    if config.spatial:
        print(f"  spatial=True, prior={config.prior}, groups={config.model.get('groups', True)}")
        print(f"  Inducing points (M): {config.model.get('num_inducing', 3000)}")
        print(f"  Kernel: {config.model.get('kernel', 'Matern32')} (lengthscale={config.model.get('lengthscale', 1.0)})")
        elbo_history, model = model.fit(
            data.Y.numpy(),
            coordinates=data.X.numpy(),
            groups=data.groups.numpy(),
            return_history=True,
        )
    else:
        elbo_history, model = model.fit(data.Y.numpy(), return_history=True)
    train_time = time.perf_counter() - t0

    max_iter = config.training.get("max_iter", 10000)
    print("\nTraining complete!")
    print(f"  Final ELBO:     {model.elbo_:.2f}")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Converged:      {model.n_iter_ < max_iter}")

    # Create model-specific output directory
    spatial = config.model.get("spatial", False)
    prior = config.model.get("prior", "GaussianPrior")
    model_name = prior.lower() if spatial else "pnmf"
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    _save_model(model, config, model_dir)
    _save_elbo_history(elbo_history, model_dir)

    metadata = _get_training_metadata(model, config, data, train_time)
    with open(model_dir / "training.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save_yaml(model_dir / "config.yaml")
    print(f"Model saved to: {model_dir}/")
