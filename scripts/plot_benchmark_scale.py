#!/usr/bin/env python
"""Plot benchmark results from outputs/benchmark_scale/results.csv."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CSV  = Path("outputs/benchmark_scale/results.csv")
OUT  = Path("outputs/benchmark_scale/benchmark_scale.png")
K_VALUES = [10, 25, 50, 75, 100]


def main():
    df = pd.read_csv(CSV)

    # Colors: distinct per M value
    m_values = sorted(df[df["model_type"] == "LCGP"]["M"].unique(), reverse=True)
    palette = {
        m_values[0]: "#1b9e77",   # teal  (M=N, full)
        m_values[1]: "#d95f02",   # orange (M=10000)
        m_values[2]: "#7570b3",   # purple (M=5000)
    } if len(m_values) == 3 else {
        m: c for m, c in zip(m_values, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])
    }

    ref     = df[df["model_type"] == "SVGP"].iloc[0]
    ref_ms  = ref["sec_per_step"] * 1e3
    ref_mem = ref["peak_memory_gb"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, y_col, y_label, ref_val, title in [
        (axes[0], "sec_per_step_ms", "ms per step",          ref_ms,  "Speed vs K"),
        (axes[1], "peak_memory_gb",  "Peak GPU memory (GB)", ref_mem, "Memory vs K"),
    ]:
        for m_val in m_values:
            subset = df[(df["model_type"] == "LCGP") & (df["M"] == m_val)].sort_values("K")
            ax.plot(
                subset["K"], subset[y_col],
                marker="o", linewidth=2.2, markersize=6,
                color=palette[m_val],
                label=f"LCGP M={int(m_val):,}",
            )

        ax.axhline(
            ref_val,
            linestyle="--", color="black", linewidth=1.5,
            label=f"SVGP M=3,000  ({ref_val:.3g})",
        )
        ax.set_xscale("log")
        ax.set_xlabel("K (neighbors)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(K_VALUES)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
