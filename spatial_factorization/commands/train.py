"""Train PNMF model (Stage 2)."""

import json
import pickle
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from PNMF import PNMF

from ..config import Config
from ..datasets.base import load_preprocessed


def _save_model(model, config: Config, model_dir: Path) -> None:
    """Save trained model to disk (both pickle and torch formats)."""
    # Pickle: Full model with sklearn API
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # PyTorch state_dict: More portable, version-independent
    torch.save({
        "model_state_dict": model._model.state_dict(),
        "prior_state_dict": model._prior.state_dict(),
        "components": model.components_,
        "hyperparameters": {
            "n_components": model.n_components,
            "mode": config.model.mode,
            "loadings_mode": config.model.loadings_mode,
            "random_state": config.seed,
        },
    }, model_dir / "model.pth")


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
        "max_iter": config.training.max_iter,
        "converged": model.n_iter_ < config.training.max_iter,
        "n_iterations": model.n_iter_,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": {
            "mode": config.model.mode,
            "loadings_mode": config.model.loadings_mode,
            "spatial": config.model.spatial,
        },
        "training_config": asdict(config.training),
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
    print(f"Loading preprocessed data from: {config.output_dir}/preprocessed/")
    data = load_preprocessed(config.output_dir)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Build model (data.Y is already in N x D format)
    print(f"Training PNMF with {config.model.n_components} components "
          f"(mode={config.model.mode}, device={config.training.device})...")

    model = PNMF(
        n_components=config.model.n_components,
        mode=config.model.mode,
        loadings_mode=config.model.loadings_mode,
        random_state=config.seed,
        **config.training.to_pnmf_kwargs(),
    )

    # Train
    t0 = time.perf_counter()
    elbo_history, model = model.fit(data.Y.numpy(), return_history=True)
    train_time = time.perf_counter() - t0

    print("\nTraining complete!")
    print(f"  Final ELBO:     {model.elbo_:.2f}")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Converged:      {model.n_iter_ < config.training.max_iter}")

    # Create model-specific output directory
    model_name = "pnmf" if not config.model.spatial else config.model.gp_class.lower()
    model_dir = config.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    _save_model(model, config, model_dir)
    _save_elbo_history(elbo_history, model_dir)

    metadata = _get_training_metadata(model, config, data, train_time)
    with open(model_dir / "training.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save_yaml(model_dir / "config.yaml")
    print(f"Model saved to: {model_dir}/")
