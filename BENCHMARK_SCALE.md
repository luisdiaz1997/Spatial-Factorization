# Scale Benchmark: SVGP vs LCGP(K)

Measures **per-step training speed** and **peak GPU memory** for SVGP and LCGP
at varying neighborhood sizes K on the real Slideseq dataset (~41K spots,
~4K genes).

## What it tests

| Configuration | Model | Complexity |
|---------------|-------|-----------|
| SVGP          | sparse variational GP, M=3000 inducing points | O(M²) |
| LCGP K=10     | locally conditioned GP, K=10 neighbors | O(NK²) |
| LCGP K=25     | — | — |
| LCGP K=50     | — (production default) | — |
| LCGP K=75     | — | — |
| LCGP K=100    | — | — |

Both model types are non-multigroup (no MGGP prefix).  
All runs use identical hyperparameters: `n_components=10`, `lengthscale=8.0`,
`batch_size=7000`, `y_batch_size=2000`, `lr=2e-3`.

## How to run

```bash
conda run -n factorization python scripts/benchmark_scale.py
```

Requires preprocessed Slideseq data to exist (`outputs/slideseq/preprocessed/`).
Run `spatial_factorization preprocess -c configs/slideseq/svgp.yaml` first if
it is missing.

Each configuration runs **60 steps** (10 warm-up discarded + 50 measured).
Expected wall time: ~3 minutes on a single GPU.

## Output

```
outputs/benchmark_scale/
├── results.csv            # per-run stats (label, K, sec_per_step, peak_memory_gb, …)
└── benchmark_scale.png    # 1×2 figure: speed (ms/step) and memory (GB) vs K
```

## Benchmark settings

| Setting | Value |
|---------|-------|
| Warm-up steps | 10 |
| Measured steps | 50 |
| Dataset | Slideseq (N≈41K, D≈4K) |
| batch_size | 3500 |
| y_batch_size | 1000 |
| n_components (L) | 10 |
| SVGP num_inducing (M) | 3000 |
| LCGP K values | 10, 25, 50, 75, 100 |
| LCGP neighbors | knn (deterministic) |
| Seed | 67 |

## Results

<!-- Updated after running the benchmark -->

![Speed and memory benchmark](outputs/benchmark_scale/benchmark_scale.png)
