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
    except (AttributeError, TypeError, pickle.PicklingError) as e:
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
        state["hyperparameters"]["multigroup"] = config.groups
        state["hyperparameters"]["local"] = config.local

    torch.save(state, model_dir / "model.pth")


def _save_elbo_history(elbo_history: list, model_dir: Path) -> None:
    """Save ELBO history (both CSV and numpy formats)."""
    df = pd.DataFrame({"iteration": range(len(elbo_history)), "elbo": elbo_history})
    df.to_csv(model_dir / "elbo_history.csv", index=False)
    np.save(model_dir / "elbo_history.npy", np.array(elbo_history))


def _append_elbo_history(new_elbo_history: list, model_dir: Path) -> None:
    """Append new ELBO values to existing elbo_history.csv/npy."""
    csv_path = model_dir / "elbo_history.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        start_iter = int(existing_df["iteration"].max()) + 1
    else:
        existing_df = pd.DataFrame({"iteration": [], "elbo": []})
        start_iter = 0

    new_df = pd.DataFrame({
        "iteration": range(start_iter, start_iter + len(new_elbo_history)),
        "elbo": new_elbo_history,
    })
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    np.save(model_dir / "elbo_history.npy", combined["elbo"].values)


def _create_warm_start_pnmf(loaded_model, config: Config, pnmf_kwargs: dict):
    """Create a PNMF subclass that warm-starts training from a loaded model state.

    Overrides _create_spatial_prior and _initialize_W (and _initialize_mu_nonspatial
    for non-spatial models) to inject the loaded parameters instead of fresh random
    initialization, then lets fit() run the training loop normally.
    """
    loaded_W = loaded_model._model.W.data.clone().cpu()

    if config.spatial:
        loaded_prior = loaded_model._prior

        class _WarmStartPNMF(PNMF):
            def _create_spatial_prior(self, Y, coordinates, groups):
                return loaded_prior

            def _initialize_W(self, X_torch):
                device = self._get_device()
                self._model.W.data = loaded_W.to(device)
                n = X_torch.shape[1]
                dummy = np.zeros((n, self.n_components))
                return dummy, dummy

        return _WarmStartPNMF(**pnmf_kwargs)

    else:
        loaded_prior_sd = {k: v.clone().cpu() for k, v in loaded_model._prior.state_dict().items()}

        class _WarmStartPNMF(PNMF):
            def _initialize_W(self, X_torch):
                device = self._get_device()
                self._model.W.data = loaded_W.to(device)
                n = X_torch.shape[1]
                dummy = np.zeros((n, self.n_components))
                return dummy, dummy

            def _initialize_mu_nonspatial(self, exp_F_init):
                device = self._get_device()
                sd = {k: v.to(device) for k, v in loaded_prior_sd.items()}
                self._prior.load_state_dict(sd)

        return _WarmStartPNMF(**pnmf_kwargs)


def _get_training_metadata(model, config: Config, data, train_time: float, prev_n_iterations: int = 0) -> dict:
    """Build training metadata dict."""
    return {
        "n_components": model.n_components_,
        "elbo": float(model.elbo_),
        "training_time": train_time,
        "max_iter": config.training.get("max_iter", 10000),
        "converged": model.n_iter_ < config.training.get("max_iter", 10000),
        "n_iterations": prev_n_iterations + model.n_iter_,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": config.model,
        "training_config": config.training,
        "data_info": {
            "n_spots": data.n_spots,
            "n_genes": data.n_genes,
            "n_groups": data.n_groups,
        },
    }


def run(config_path: str, resume: bool = False):
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

    # Determine model output directory early (needed for resume)
    model_dir = output_dir / config.model_name

    # Resume: load existing model and warm-start from its parameters
    prev_n_iterations = 0
    if resume:
        pth_path = model_dir / "model.pth"
        if not pth_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {pth_path}. "
                "Cannot resume. Run without --resume to train from scratch."
            )
        print(f"Resuming training from: {model_dir}/")
        from .analyze import _load_model
        loaded_model = _load_model(model_dir)

        # Track previous iteration count for cumulative metadata
        training_json = model_dir / "training.json"
        if training_json.exists():
            with open(training_json) as f:
                prev_meta = json.load(f)
            prev_n_iterations = prev_meta.get("n_iterations", 0)
        print(f"  Previous iterations: {prev_n_iterations}")

        model = _create_warm_start_pnmf(loaded_model, config, config.to_pnmf_kwargs())
        print(f"Continuing PNMF with {n_components} components (mode={mode}, device={device})...")
    else:
        model = PNMF(random_state=config.seed, **config.to_pnmf_kwargs())
        print(f"Training PNMF with {n_components} components (mode={mode}, device={device})...")

    # Train (spatial models require coordinates; groups are optional)
    t0 = time.perf_counter()
    if config.spatial:
        print(f"  spatial=True, prior={config.prior}, groups={config.groups}, local={config.local}")
        if config.local:
            K = config.model.get('K', 50)
            print(f"  LCGP: K={K}")
        else:
            print(f"  Inducing points (M): {config.model.get('num_inducing', 3000)}")
        print(f"  Kernel: {config.model.get('kernel', 'Matern32')} (lengthscale={config.model.get('lengthscale', 1.0)})")

        fit_kwargs = dict(
            coordinates=data.X.numpy(),
            return_history=True,
        )
        if config.groups:
            fit_kwargs["groups"] = data.groups.numpy()

        elbo_history, model = model.fit(data.Y.numpy(), **fit_kwargs)
    else:
        elbo_history, model = model.fit(data.Y.numpy(), return_history=True)
    train_time = time.perf_counter() - t0

    max_iter = config.training.get("max_iter", 10000)
    print("\nTraining complete!")
    print(f"  Final ELBO:     {model.elbo_:.2f}")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Converged:      {model.n_iter_ < max_iter}")

    # Save outputs
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_model(model, config, model_dir)

    if resume:
        _append_elbo_history(elbo_history, model_dir)
    else:
        _save_elbo_history(elbo_history, model_dir)

    metadata = _get_training_metadata(model, config, data, train_time, prev_n_iterations)
    if resume:
        metadata["resumed"] = True
    with open(model_dir / "training.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save_yaml(model_dir / "config.yaml")
    print(f"Model saved to: {model_dir}/")
