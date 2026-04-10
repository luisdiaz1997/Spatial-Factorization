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
        if config.local:
            state["hyperparameters"]["K"] = config.model.get("K", 50)
            state["hyperparameters"]["neighbors"] = config.model.get("neighbors", "knn")

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

        # Replicate PNMF's parameter freezing that normally happens inside
        # _create_spatial_prior (models.py lines 686-757). When loading from
        # .pth, requires_grad flags are not preserved (state dicts save values
        # only), so we must re-apply them here before the optimizer is built.
        loaded_prior.Z.requires_grad_(False)
        if hasattr(loaded_prior, 'groupsZ') and isinstance(loaded_prior.groupsZ, torch.nn.Parameter):
            loaded_prior.groupsZ.requires_grad_(False)
        loaded_prior.kernel.sigma.requires_grad_(False)
        if not config.training.get("train_lengthscale", False):
            loaded_prior.kernel.lengthscale.requires_grad_(False)
        if hasattr(loaded_prior.kernel, 'group_diff_param'):
            loaded_prior.kernel.group_diff_param.requires_grad_(False)

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


def _clamp_data_dims(kwargs: dict, N: int, D: int) -> dict:
    """Clamp num_inducing, batch_size, y_batch_size to actual data dimensions."""
    if kwargs.get("num_inducing") is not None:
        clamped = min(kwargs["num_inducing"], N)
        if clamped < kwargs["num_inducing"]:
            print(f"  Note: num_inducing clamped {kwargs['num_inducing']} → {clamped} (N={N})")
        kwargs["num_inducing"] = clamped

    if kwargs.get("batch_size") is not None:
        clamped = min(kwargs["batch_size"], N)
        if clamped < kwargs["batch_size"]:
            print(f"  Note: batch_size clamped {kwargs['batch_size']} → {clamped} (N={N})")
        kwargs["batch_size"] = clamped

    if kwargs.get("y_batch_size") is not None:
        clamped = min(kwargs["y_batch_size"], D)
        if clamped < kwargs["y_batch_size"]:
            print(f"  Note: y_batch_size clamped {kwargs['y_batch_size']} → {clamped} (D={D})")
        kwargs["y_batch_size"] = clamped

    return kwargs


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


def _make_video_callback(config: Config, data, video_interval: int, frames: list, frame_iters: list):
    """Return a callback that captures factor snapshots during training."""
    import torch

    def callback(model, iteration, elbo_value):
        with torch.no_grad():
            if config.spatial:
                # Run GP predictive pass on all training coords (batched to avoid OOM)
                from PNMF.transforms import _get_spatial_qF
                coords_np = data.X.numpy()
                groups_np = data.groups.numpy() if config.groups else None
                chunk = 10000
                N = coords_np.shape[0]
                chunks_F = []
                for start in range(0, N, chunk):
                    coords_b = torch.from_numpy(coords_np[start:start + chunk]).to(model._get_device())
                    groups_b = (
                        torch.from_numpy(groups_np[start:start + chunk]).to(model._get_device())
                        if groups_np is not None else None
                    )
                    qF = _get_spatial_qF(model, coordinates=coords_b, groups=groups_b)
                    chunks_F.append(torch.exp(qF.mean.detach().cpu()).T)  # (chunk, L)
                F = torch.cat(chunks_F, dim=0).numpy()  # (N, L)
            else:
                # Non-spatial: exp(prior.mean) transposed
                F = torch.exp(model._prior.mean.detach().cpu()).T.numpy()  # (N, L)

        frames.append(F)
        frame_iters.append(iteration)

    return callback



def run(config_path: str, resume: bool = False, video: bool = False, probabilistic: bool = False):
    """Train a PNMF model from config.

    Output files (outputs/{dataset}/{model}/):
        model.pkl          - Trained PNMF model (pickle)
        model.pth          - PyTorch state dict
        training.json      - Training metadata
        elbo_history.csv   - ELBO history
        config.yaml        - Copy of config used
    """
    config = Config.from_yaml(config_path)

    # --probabilistic overrides the KNN strategy for LCGP training. Mutate the
    # config so to_pnmf_kwargs passes neighbors=probabilistic to PNMF and the
    # saved hyperparameters reflect the strategy actually used.
    if probabilistic:
        if config.local:
            config.model["neighbors"] = "probabilistic"
            print("[--probabilistic] Overriding KNN strategy to 'probabilistic' for training")
        else:
            print("[--probabilistic] Ignored: only affects LCGP (local=True) training")

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

    # Clamp model/training params to actual data dimensions
    pnmf_kwargs = _clamp_data_dims(config.to_pnmf_kwargs(), data.n_spots, data.n_genes)

    # Determine model output directory early (needed for resume)
    model_dir = output_dir / config.model_name

    # Resume: load existing model and warm-start from its parameters
    prev_n_iterations = 0
    if resume:
        pth_path = model_dir / "model.pth"
        if not pth_path.exists():
            print(f"  Note: no checkpoint at {pth_path}, training from scratch.")
            resume = False

    if resume:
        print(f"Resuming training from: {model_dir}/")
        from .analyze import _load_model
        loaded_model = _load_model(model_dir, neighbors_override="probabilistic" if probabilistic else None)

        # Track previous iteration count for cumulative metadata
        training_json = model_dir / "training.json"
        if training_json.exists():
            with open(training_json) as f:
                prev_meta = json.load(f)
            prev_n_iterations = prev_meta.get("n_iterations", 0)
        print(f"  Previous iterations: {prev_n_iterations}")

        model = _create_warm_start_pnmf(loaded_model, config, pnmf_kwargs)
        print(f"Continuing PNMF with {n_components} components (mode={mode}, device={device})...")
    else:
        model = PNMF(random_state=config.seed, **pnmf_kwargs)
        print(f"Training PNMF with {n_components} components (mode={mode}, device={device})...")

    # Build optional video callback
    video_frames = []
    video_frame_iters = []
    video_interval = config.training.get("video_interval", 20)
    if video:
        print(f"  Video mode: capturing frames every {video_interval} iterations")
        video_callback = _make_video_callback(config, data, video_interval, video_frames, video_frame_iters)
    else:
        video_callback = None

    # Train (spatial models require coordinates; groups are optional)
    t0 = time.perf_counter()
    if config.spatial:
        print(f"  spatial=True, prior={config.prior}, groups={config.groups}, local={config.local}")
        if config.local:
            K = config.model.get('K', 50)
            neighbors = config.model.get('neighbors', 'knn')
            print(f"  LCGP: K={K}, neighbors={neighbors}")
        else:
            print(f"  Inducing points (M): {pnmf_kwargs.get('num_inducing', 3000)}")
        print(f"  Kernel: {config.model.get('kernel', 'Matern32')} (lengthscale={config.model.get('lengthscale', 1.0)})")

        fit_kwargs = dict(
            coordinates=data.X.numpy(),
            return_history=True,
        )
        if config.groups:
            fit_kwargs["groups"] = data.groups.numpy()
        if video_callback is not None:
            fit_kwargs["callback"] = video_callback
            fit_kwargs["callback_interval"] = video_interval

        elbo_history, model = model.fit(data.Y.numpy(), **fit_kwargs)
    else:
        fit_kwargs = dict(return_history=True)
        if video_callback is not None:
            fit_kwargs["callback"] = video_callback
            fit_kwargs["callback_interval"] = video_interval
        elbo_history, model = model.fit(data.Y.numpy(), **fit_kwargs)
    train_time = time.perf_counter() - t0

    if resume:
        # _WarmStartPNMF is a local class that can't be pickled. Training is
        # complete so the overrides are no longer needed; restore the base class
        # so _save_model can pickle the model identically to a normal train run.
        model.__class__ = PNMF

    max_iter = config.training.get("max_iter", 10000)
    print("\nTraining complete!")
    print(f"  Final ELBO:     {model.elbo_:.2f}")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Converged:      {model.n_iter_ < max_iter}")

    # Save outputs
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_model(model, config, model_dir)

    # Save video frames if captured (analyze will reorder, figures will render GIF)
    if video and video_frames:
        frames_arr = np.stack(video_frames, axis=0)  # (n_frames, N, L)
        np.save(model_dir / "video_frames.npy", frames_arr)
        np.save(model_dir / "video_frame_iters.npy", np.array(video_frame_iters))
        print(f"\nSaved {len(video_frames)} video frames to {model_dir}/video_frames.npy")

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
