# Training Animation Videos

Videos that illustrate the effect of proper KL / likelihood scaling during
variational inference, using the Slide-seq V2 dataset as a reference.

---

## Motivation

The ELBO has two terms that must be balanced:

```
ELBO = E_q[log p(Y | F)]  -  KL[q(F) || p(F)]
         likelihood              regularization
```

Both terms need to be estimated on mini-batches and then scaled to approximate
the full-data ELBO. Getting the scaling wrong creates either an over-regularized
or under-regularized model, and the training animations make this directly
visible as factor maps emerge (or fail to emerge) over iterations.

---

## Two Scaling Problems

### 1 · PNMF — D-batch scaling of the likelihood

When we sub-sample genes (y-batch size D_batch < D total genes), the observed
log-likelihood is computed on only D_batch / D of the data.  The KL term,
however, is over the full set of cells and does not change.

| Regime | Scaling applied | Effect |
|--------|----------------|--------|
| **Unscaled** (old) | LL estimated on D_batch genes, *not* scaled up | LL ≪ KL → **over-regularized**: factors collapse, no structure emerges |
| **Scaled** (new) | LL × (D / D_batch) | LL and KL are commensurate → factors develop clear spatial patterns |

Concretely, for Slide-seq (D = 4,018 genes, D_batch = 2,000):
```
LL scaling factor = D / D_batch = 4018 / 2000 ≈ 2×
```
Without this, the optimiser sees a KL that is 2× too large relative to the
likelihood and pushes the variational posteriors toward the prior, washing out
any learned structure.

### 2 · SVGP — N/M scaling of the KL

The SVGP KL is over M inducing points (e.g. M = 3,000), while the likelihood
is summed over N training cells (e.g. N = 41,786 for Slide-seq).  The two
terms live on very different scales.

| Regime | Scaling applied | Effect |
|--------|----------------|--------|
| **Unscaled** (old) | KL over M inducing points, *no* N/M boost | KL ≪ LL → **under-regularized**: factors are noisy, inducing point posterior drifts without constraint |
| **Scaled** (new) | KL × (N / M) | GP prior is properly enforced → smooth, spatially coherent factors |

Concretely, for Slide-seq (N = 41,786, M = 3,000):
```
KL scaling factor = N / M = 41786 / 3000 ≈ 13.9×
```
Without this, the inducing-point KL is ~14× too small, and the model is
essentially unconstrained by the GP prior, producing spatially incoherent
factors.

---

## Video Configs (`configs/slideseq/video/`)

Four YAML configs are created in `configs/slideseq/video/`, all pointing to a
separate output directory `video_outputs/slideseq/` so that no existing trained
models are touched.

```
configs/slideseq/video/
├── pnmf_unscaled.yaml    # PNMF, scale_ll_D: false  (old behaviour)
├── pnmf_scaled.yaml      # PNMF, scale_ll_D: true   (correct)
├── svgp_unscaled.yaml    # SVGP, scale_kl_NM: false (old behaviour)
└── svgp_scaled.yaml      # SVGP, scale_kl_NM: true  (correct)
```

### Key config parameters

```yaml
# --- PNMF variants ---
training:
  max_iter: 2000          # enough iterations to see convergence dynamics
  video_interval: 20      # capture a frame every 20 iterations → ~100 frames

model:
  scale_ll_D: true/false  # new flag: whether to scale LL by D/D_batch

# --- SVGP variants ---
training:
  max_iter: 500
  video_interval: 10      # capture a frame every 10 iterations → ~50 frames

model:
  scale_kl_NM: true/false # new flag: whether to scale SVGP KL by N/M
```

Both variants share the same preprocessing parameters, dataset, and
hyper-parameters as `configs/slideseq/general.yaml` (n_components=10,
lengthscale=8.0, batch_size=7000, y_batch_size=2000, etc.) so that the only
difference is the scaling flag.

### Shared preprocessing

All four configs share the same `output_dir: video_outputs/slideseq` so that
`preprocess` only needs to be run once:

```bash
spatial_factorization preprocess -c configs/slideseq/video/pnmf_scaled.yaml
```

---

## `--video` Flag (implementation plan)

```bash
spatial_factorization train --video -c configs/slideseq/video/pnmf_scaled.yaml
```

When `--video` is passed, the train command:

1. Registers a **callback** with the PNMF `fit()` call that fires every
   `training.video_interval` iterations.
2. Inside the callback, grabs the current factor map:
   - **PNMF / LCGP**: `exp(model._prior.mean)`, shape `(L, N)` → `(N, L)`
   - **SVGP / MGGP_SVGP**: runs a full GP predictive pass on all training
     coordinates to obtain `exp(qF.mean)` at all N points (more expensive but
     only done every `video_interval` steps).
3. Appends the `(N, L)` array and the current iteration number to an in-memory
   list of frames.
4. After training completes, saves:
   - `{model_dir}/video_frames.npy` — shape `(n_frames, N, L)` for reuse
   - `{model_dir}/video_frame_iters.npy` — iteration number for each frame
   - `{model_dir}/training_animation.mp4` — rendered video (fallback: `.gif`)

### Video layout

The animation shows the **top-4 factors by final Moran's I** (same ordering
used by the analyze stage), with each factor in one column:

```
[ Factor 0 ] [ Factor 1 ] [ Factor 2 ] [ Factor 3 ]
  iter=0        iter=0        iter=0        iter=0
```

Each frame updates all four scatter plots simultaneously. The title shows the
current iteration and instantaneous ELBO.

### FuncAnimation pattern (from `nnnsf_visium_anim_experiment.ipynb`)

```python
from matplotlib.animation import FuncAnimation, FFMpegWriter

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

def update(frame_idx):
    F = frames[frame_idx]              # (N, 4) — top 4 factors
    for i, ax in enumerate(axes):
        ax.cla()
        ax.scatter(X[:, 0], X[:, 1], c=F[:, i], s=s, cmap='turbo')
        ax.set_title(f'Factor {i+1} | iter {frame_iters[frame_idx]}')
        ax.axis('off')

anim = FuncAnimation(fig, update, frames=range(len(frames)), interval=100)
writer = FFMpegWriter(fps=10, bitrate=2000, codec='h264')
anim.save(model_dir / 'training_animation.mp4', writer=writer, dpi=100)
```

Point size follows the same auto-scaling rule as `figures.py`:
`s = 100 / sqrt(N)` → s ≈ 0.49 for Slide-seq N = 41,786.

---

## PNMF changes needed (`Probabilistic-NMF/PNMF/models.py`)

### New constructor parameters

```python
# KL / likelihood scaling flags
scale_ll_D: bool = True    # scale LL by D/D_batch when y-batching (default: True = current correct behaviour)
scale_kl_NM: bool = True   # scale SVGP KL by N/M (default: True = current correct behaviour)
```

These default to `True` so **all existing runs are unaffected**.  Only the
video configs set them to `False` to reproduce the old (broken) behaviour.

### Training loop changes

```python
# Non-spatial and spatial LL scaling (both cases):
if self.y_batch_size is not None and self.scale_ll_D:          # ← add `and self.scale_ll_D`
    exp_ll = exp_ll * (D / min(self.y_batch_size, D))

# SVGP KL scaling:
if self.scale_kl_NM:                                           # ← wrap in flag
    kl = self._prior.kl_divergence(qU, pU).sum() * (N / M)
else:
    kl = self._prior.kl_divergence(qU, pU).sum()
```

### Callback in `fit()`

```python
def fit(self, X, y=None, coordinates=None, groups=None,
        return_history=False,
        callback=None,            # callable(model, iteration, elbo) or None
        callback_interval=100):   # fire callback every N iterations
    ...
    for iteration in pbar:
        ...
        if callback is not None and iteration % callback_interval == 0:
            callback(self, iteration, elbo_value)
```

### Passing through `config.to_pnmf_kwargs()` (`spatial_factorization/config.py`)

```python
kwargs["scale_ll_D"]  = self.model.get("scale_ll_D", True)
kwargs["scale_kl_NM"] = self.model.get("scale_kl_NM", True)
```

---

## CLI change (`spatial_factorization/cli.py`)

```python
@cli.command()
@click.option("--config", "-c", required=True, ...)
@click.option("--resume", is_flag=True, ...)
@click.option("--video",  is_flag=True, default=False,
              help="Capture factor snapshots during training and save as MP4.")
def train(config, resume, video):
    from .commands import train as train_cmd
    train_cmd.run(config, resume=resume, video=video)
```

---

## Comparison video (post-processing)

After training all four variants the frames can be combined into a side-by-side
comparison using the saved `video_frames.npy` files.  Each comparison shows
two columns: unscaled (left) and scaled (right), sharing the same colour scale
and the same factor index so the difference is immediately apparent.

```
PNMF comparison: pnmf_unscaled vs pnmf_scaled
SVGP comparison: svgp_unscaled vs svgp_scaled
```

A `make_comparison_video` helper (to be added in `train.py` or a standalone
script) loads both `video_frames.npy` files, aligns them by iteration number,
and renders the comparison MP4.

---

## Expected outputs

```
video_outputs/slideseq/
├── preprocessed/               # shared (preprocess once)
│
├── pnmf_unscaled/pnmf/
│   ├── training_animation.mp4  # over-regularized: flat/noisy factors
│   ├── video_frames.npy        # (n_frames, N, L)
│   └── video_frame_iters.npy   # (n_frames,)
│
├── pnmf_scaled/pnmf/
│   ├── training_animation.mp4  # correct: clear spatial patterns emerge
│   ├── video_frames.npy
│   └── video_frame_iters.npy
│
├── svgp_unscaled/svgp/
│   ├── training_animation.mp4  # under-regularized: noisy, incoherent GP
│   ├── video_frames.npy
│   └── video_frame_iters.npy
│
└── svgp_scaled/svgp/
    ├── training_animation.mp4  # correct: smooth, spatially structured
    ├── video_frames.npy
    └── video_frame_iters.npy
```

---

## Running the video pipeline

```bash
# 1. Preprocess once (all four configs share the same dataset/params)
spatial_factorization preprocess -c configs/slideseq/video/pnmf_scaled.yaml

# 2. Train + capture frames
spatial_factorization train --video -c configs/slideseq/video/pnmf_unscaled.yaml
spatial_factorization train --video -c configs/slideseq/video/pnmf_scaled.yaml
spatial_factorization train --video -c configs/slideseq/video/svgp_unscaled.yaml
spatial_factorization train --video -c configs/slideseq/video/svgp_scaled.yaml
```

---

## Notes on animation approach

The animation code follows the pattern from
`GPzoo/notebooks/nnnsf_visium_anim_experiment.ipynb`:

- Frames are captured **in memory** during training (no disk I/O per frame).
- The `update(frame_idx)` function clears and redraws all axes each frame
  (`ax.cla()`) to avoid matplotlib artist accumulation.
- `FuncAnimation` is used with `writer="pillow"` for GIF or `FFMpegWriter`
  for MP4 (preferred — smaller file, better quality).
- The `interval` parameter (milliseconds) controls display speed in notebooks;
  the `fps` parameter to `anim.save()` controls the saved video frame rate.
- For Slide-seq (N ≈ 41K), `rasterized=True` on the scatter keeps the MP4
  file size manageable.
