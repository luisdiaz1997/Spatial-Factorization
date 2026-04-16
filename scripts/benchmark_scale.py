#!/usr/bin/env python
"""
Speed and memory benchmark: SVGP vs LCGP(K) on the Slideseq dataset.

For each configuration (SVGP + LCGP at K ∈ {10, 25, 50, 75, 100}) we run
N_STEPS training steps (after N_WARMUP warm-up steps), recording:
  - per-step wall-clock time (ms)
  - peak GPU memory (GB)

Trained models are NOT saved. Output:
  outputs/benchmark_scale/results.csv
  outputs/benchmark_scale/benchmark_scale.png
"""

import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PNMF import PNMF
from spatial_factorization.datasets.base import load_preprocessed

# ── Benchmark settings ────────────────────────────────────────────────────────
DATA_DIR      = Path("outputs/slideseq")
OUT_DIR       = Path("outputs/benchmark_scale")
N_WARMUP      = 10     # steps discarded before timing starts
N_STEPS       = 50     # steps used for the per-step statistics
SEED          = 67
BATCH_SIZE    = 3500
Y_BATCH_SIZE  = 3000
LEARNING_RATE = 2e-3
K_VALUES      = [10, 25, 50, 75, 100]
N_COMPONENTS  = 10
# ─────────────────────────────────────────────────────────────────────────────


def _make_timing_callback(n_warmup, step_times):
    """Callback that records wall-clock time for steps after n_warmup."""
    state = {"count": 0, "last_t": None}

    def cb(model, iteration, elbo_value):
        t = time.perf_counter()
        if state["last_t"] is not None and state["count"] >= n_warmup:
            step_times.append(t - state["last_t"])
        state["last_t"] = t
        state["count"] += 1

    return cb


def _run_one(label, extra_kwargs, Y_np, X_np):
    """
    Run a single benchmark configuration.

    Parameters
    ----------
    label       : display label
    extra_kwargs: kwargs added on top of the common set (model-specific)
    Y_np        : (D, N) count array
    X_np        : (N, 2) spatial coords

    Returns
    -------
    dict with timing and memory stats
    """
    total_steps = N_WARMUP + N_STEPS

    common = dict(
        n_components   = N_COMPONENTS,
        mode           = "expanded",
        loadings_mode  = "multiplicative",
        training_mode  = "standard",
        E              = 3,
        scale_ll_D     = True,
        scale_kl_NM    = True,
        max_iter       = total_steps,
        tol            = 1e-12,        # disable early stopping
        verbose        = False,
        batch_size     = BATCH_SIZE,
        y_batch_size   = Y_BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        optimizer      = "Adam",
        shuffle        = True,
        device         = "auto",
        random_state   = SEED,
    )
    kwargs = {**common, **extra_kwargs}

    print(f"\n── {label} {'─' * max(0, 50 - len(label))}")
    print("   Building model …", flush=True)
    model = PNMF(**kwargs)

    step_times = []
    cb = _make_timing_callback(N_WARMUP, step_times)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(f"   Running {total_steps} steps "
          f"(warmup={N_WARMUP}, measured={N_STEPS}) ...", flush=True)

    _, model = model.fit(
        Y_np,
        coordinates=X_np,
        return_history=True,
        callback=cb,
        callback_interval=1,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024 ** 3
        if torch.cuda.is_available()
        else float("nan")
    )

    mean_ms = np.mean(step_times) * 1e3
    std_ms  = np.std(step_times)  * 1e3
    print(f"   {mean_ms:.1f} ± {std_ms:.1f} ms/step  |  peak GPU: {peak_gb:.3f} GB",
          flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label":             label,
        "sec_per_step":      float(np.mean(step_times)),
        "sec_std":           float(np.std(step_times)),
        "peak_memory_gb":    peak_gb,
        "n_measured_steps":  len(step_times),
    }


def _plot(df, out_path):
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ref     = df[df["label"] == "SVGP"].iloc[0]
    ref_ms  = ref["sec_per_step"] * 1e3
    ref_mem = ref["peak_memory_gb"]

    lcgp = df[df["K"].notna()].copy().sort_values("K")

    for ax, y_col, y_label, ref_val, title in [
        (axes[0], "sec_per_step_ms", "ms per step",       ref_ms,  "Speed vs K"),
        (axes[1], "peak_memory_gb",  "Peak GPU memory (GB)", ref_mem, "Memory vs K"),
    ]:
        ax.plot(lcgp["K"], lcgp[y_col], marker="o", linewidth=2, label="LCGP(K)")
        ax.axhline(
            ref_val,
            linestyle="--", color="black", linewidth=1.5,
            label=f"SVGP  ({ref_val:.3g})",
        )
        ax.set_xscale("log")
        ax.set_xlabel("K (LCGP neighbors)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        k_ticks = lcgp["K"].astype(int).tolist()
        ax.set_xticks(k_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", rotation=0)
        ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed Slideseq data …")
    data = load_preprocessed(DATA_DIR)
    Y_np = data.Y.numpy()   # (D, N)
    X_np = data.X.numpy()   # (N, 2)
    print(f"  N = {data.n_spots:,}   D = {data.n_genes:,}")

    # ── Model configurations ──────────────────────────────────────────────────
    svgp_kw = dict(
        spatial            = True,
        multigroup         = False,
        local              = False,
        kernel             = "Matern32",
        num_inducing       = 3000,
        lengthscale        = 8.0,
        sigma              = 1.0,
        train_lengthscale  = False,
        cholesky_mode      = "exp",
        diagonal_only      = False,
        inducing_allocation= "proportional",
        group_diff_param   = 1.0,
    )

    runs = [("SVGP", svgp_kw, None)]

    for K in K_VALUES:
        lcgp_kw = dict(
            spatial            = True,
            multigroup         = False,
            local              = True,
            kernel             = "Matern32",
            K                  = K,
            precompute_knn     = True,
            neighbors          = "knn",
            lengthscale        = 8.0,
            sigma              = 1.0,
            train_lengthscale  = False,
            group_diff_param   = 1.0,
        )
        runs.append((f"LCGP K={K}", lcgp_kw, K))

    # ── Run ───────────────────────────────────────────────────────────────────
    results = []
    for label, kw, k_val in runs:
        stats = _run_one(label, kw, Y_np, X_np)
        stats["K"] = k_val
        results.append(stats)

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df["sec_per_step_ms"] = df["sec_per_step"] * 1e3

    csv_path = OUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    print(
        df[["label", "K", "sec_per_step_ms", "peak_memory_gb", "n_measured_steps"]]
        .to_string(index=False)
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(df, OUT_DIR / "benchmark_scale.png")


if __name__ == "__main__":
    main()
