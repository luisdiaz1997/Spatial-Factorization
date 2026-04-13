# Multiplex Arbitrary Stage Combinations

## Goal

Enable the live multiplexer (parallel GPU/CPU scheduling + rich status table) for any
stage sequence, not just `run all`. Specifically we need:

```bash
# Re-run analyze+figures for all per-sample liver configs in parallel
spatial_factorization run analyze figures -c configs/liver/ --config-name general.yaml
```

Today `run all` routes to `JobRunner`. Any other stage list runs sequentially for a
single config. We want `JobRunner` to kick in whenever config is a directory or
`general.yaml`, regardless of which stages are requested.

---

## Current Architecture

```
run all     → JobRunner.run()
               Phase 1: resolve per-model configs (from dir / general.yaml)
               Phase 2: preprocess once per dataset
               Phase 3: create Job objects + status rows (train + analyze per job)
               Phase 4: _run_parallel()
                  training loop  → launches "train" subprocess per job
                  analyze loop   → launches "analyze figures" subprocess after train completes
               Phase 5: print report + save run_status.json

run <stages> → sequential loop over ordered stages for a single config
               (no multiplexer, no status table)
```

`_run_parallel` is hardcoded to two phases:
- **Train phase**: all jobs start as "pending", get GPU/CPU, run `run train -c <cfg>`
- **Analyze phase**: jobs enter analyze queue only after their train completes, run `run analyze figures -c <cfg>`

---

## Proposed Changes

### 1. Add `stages` parameter to `JobRunner`

```python
class JobRunner:
    def __init__(
        self,
        config_path,
        stages: List[str] = None,   # NEW — defaults to full pipeline
        force_preprocess=False,
        dry_run=False,
        resume=False,
        config_name="general.yaml",
        failed_only=False,
        video=False,
        gpu_only=False,
    ):
        self.stages = stages or ["train", "analyze", "figures"]
        ...
```

`stages` is the ordered list of pipeline stages to run for each job. The runner
derives its behavior from this list rather than hardcoding train→analyze.

### 2. Generalize `_run_parallel` with two modes

**Mode A — train included** (`"train" in self.stages`): current behavior.
- Train phase: launch `run train` per job
- Analyze phase: launch `run <remaining stages>` after train finishes

**Mode B — train not included** (`"train" not in self.stages`): no training.
- Skip training phase entirely
- All jobs go directly into the analyze queue (no dependency to wait on)
- Launch `run <self.stages> -c <cfg>` immediately as resources become available

This means the "analyze loop" in `_run_parallel` stays almost identical —
we just pre-populate `analyze_pending` with all job names and set
`training_done = {all job names}` before entering the loop.

The stages passed to `_launch_process` for the "analyze" task become
`self.stages` (e.g. `["analyze", "figures"]` or `["figures"]`).

### 3. Status rows: only add rows for stages that will run

Current: always adds `{job}_train` + `{job}_analyze` rows.

Proposed: conditionally add rows based on `self.stages`:
- `"train" in stages` → add train row
- at least one non-train stage in stages → add analyze row

For Mode B (no train), each job gets only one row (the "analyze" row), labeled
with the actual stages being run (e.g. task=`"analyze"` or task=`"figures"`).

The `task` field on `JobStatus` can stay as `"analyze"` for any non-train work
since the status display already uses it as a catch-all for post-train stages.

### 4. Skip preprocessing in Mode B

Preprocessing should only run when `"preprocess" in self.stages`. In Mode B we
skip Phase 2 entirely (data already exists).

### 5. Extend `cli.py` `run_pipeline` to invoke multiplexer for non-"all" stages

```python
# In run_pipeline():

stages = [s.lower() for s in stages]

if "all" in stages:
    JobRunner(config, stages=["train","analyze","figures"], ...).run()
    return

# NEW: if config is a directory or general.yaml, route non-"all" stages
# through the multiplexer too
config_path = Path(config)
if config_path.is_dir() or Config.is_general_config(config_path):
    ordered = [s for s in STAGE_ORDER if s in stages]
    JobRunner(config, stages=ordered, ...).run()
    return

# Existing sequential path (single per-model config, no multiplexer)
...
```

This way `run analyze figures -c configs/liver/` fans out to all per-model
configs found under `configs/liver/` and runs them in parallel.

---

## Resulting CLI Behavior

| Command | Config type | Behavior |
|---------|-------------|----------|
| `run all -c general.yaml` | general | current: train→analyze+figures (parallel) |
| `run all -c configs/liver/` | dir | current: train→analyze+figures (all samples, parallel) |
| `run analyze figures -c general.yaml` | general | NEW: analyze+figures only (parallel, no train) |
| `run analyze figures -c configs/liver/` | dir | NEW: analyze+figures for all samples (parallel) |
| `run figures -c configs/liver/` | dir | NEW: figures only for all samples (parallel) |
| `run analyze figures -c svgp.yaml` | per-model | existing sequential path (single job) |
| `run train analyze -c general.yaml` | general | NEW: train then analyze (no figures, parallel) |

---

## Status Table in Mode B

For `run analyze figures -c configs/liver/`:

```
                              Analyze+Figures Progress
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Job                           ┃ Task      ┃ Device   ┃ Status     ┃ Time             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ liver_diseased_AM031_mggp_lcgp │ analyze  │ cuda:0   │ analyzing  │ 0:00:12/-        │
│ liver_diseased_AM031_svgp      │ analyze  │ cuda:1   │ analyzing  │ 0:00:08/-        │
│ liver_diseased_AM031_pnmf      │ analyze  │ cpu      │ analyzing  │ 0:00:05/-        │
│ liver_diseased_AM042_mggp_lcgp │ analyze  │ pending  │ pending    │ -                │
│ ...                            │ ...      │ ...      │ ...        │ -                │
└────────────────────────────────┴──────────┴──────────┴────────────┴──────────────────┘
```

No "train" rows, no "Epoch" column (since there's no training progress to show).
The "Epoch" column can be hidden when `"train" not in self.stages`.

---

## Example Commands

```bash
# Test with one sample (5 models in parallel via multiplexer)
spatial_factorization run analyze figures -c configs/liver/diseased/AM031/general.yaml

# All per-sample + pooled configs at once (picks up all general.yaml under configs/liver/)
# Re-running pooled ones is harmless — they already have figures
spatial_factorization run analyze figures -c configs/liver/ --config-name general.yaml
```

---

## Implementation Plan

1. **`runner.py`**:
   - Add `stages: List[str]` param to `JobRunner.__init__`
   - In `run()`: skip Phase 2 (preprocess) if `"preprocess" not in self.stages`
   - In Phase 3: conditionally add train/analyze status rows based on `self.stages`
   - In `_run_parallel()`: detect Mode A vs Mode B from `"train" in self.stages`
     - Mode B: pre-populate `training_done` and `analyze_pending` with all job names,
       set `train_cpu_slot = False`, skip the training loop entirely
   - `_launch_process` for analyze task: use `non_train_stages` derived from `self.stages`

2. **`cli.py`**:
   - In `run_pipeline()`: after `"all"` check, add directory/general.yaml detection
     to route through `JobRunner` with explicit `stages=ordered`
   - Pass `stages` to `JobRunner`

3. **`status.py`**:
   - `JobStatus` already has `total_epochs`; setting it to 0 suppresses the epoch
     display. No changes needed there.
   - Optionally: hide "Epoch" and "ELBO" columns from the table header when all
     jobs have `total_epochs == 0` (cosmetic improvement).

---

## Files to Change

- `spatial_factorization/runner.py` — main changes
- `spatial_factorization/cli.py` — route non-"all" multiplexable calls through JobRunner
- `spatial_factorization/status.py` — optional: hide epoch/ELBO columns in Mode B
