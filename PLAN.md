# Sherlock SLURM Integration Plan

## Goal

Submit one SLURM job per per-model config on Stanford Sherlock (~95 jobs total:
19 `general.yaml` files × 5 models). Each job runs:

```bash
spatial_factorization run all -c configs/<dataset>/svgp.yaml --resume
```

The existing multiplexer handles the single-model pipeline (preprocess check →
train → analyze → figures) within the job.

---

## Files to Change

| File | Action |
|------|--------|
| `spatial_factorization/runner.py` | Fix `query_gpus()` to respect `CUDA_VISIBLE_DEVICES` |
| `scripts/submit_sherlock.py` | New: generate configs, preprocess locally, submit jobs |

---

## Fix 1 — `runner.py`: Respect `CUDA_VISIBLE_DEVICES` in `query_gpus()`

`nvidia-smi` reports all GPUs on the node regardless of `CUDA_VISIBLE_DEVICES`.
When SLURM allocates 1 GPU from a 4-GPU node, the runner would detect all 4 and
try to use non-allocated ones — interfering with other users.

Add to `query_gpus()` (line 42) after building the list from nvidia-smi:

```python
# Respect CUDA_VISIBLE_DEVICES if set (e.g., by SLURM)
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cuda_visible and cuda_visible not in ("NoDevFiles", "-1"):
    visible_ids = {int(x) for x in cuda_visible.split(",")}
    gpus = [g for g in gpus if g.device_id in visible_ids]
```

Local runs (no `CUDA_VISIBLE_DEVICES` set): behavior unchanged.

---

## Fix 2 — `scripts/submit_sherlock.py`

### Workflow

1. **Discover** all `{config_name}` files recursively under the given path
2. **Generate** per-model configs by running `spatial_factorization generate -c <general.yaml>`
   for each general config (produces pnmf/svgp/mggp_svgp/lcgp/mggp_lcgp YAMLs)
3. **Preprocess** all unique datasets locally on the login node (sequential, CPU-only,
   fast) — avoids race condition if 5 model jobs for the same dataset all start and
   simultaneously try to write `preprocessed/Y.npz`
4. **Submit** one `sbatch` job per per-model config
5. **Print** summary table: job name | config | time | mem | SLURM job ID

### CLI

```bash
# Dry run — see all ~95 jobs with resource specs
python scripts/submit_sherlock.py -c configs/ --dry-run

# Submit test configs (10 epochs, fast)
python scripts/submit_sherlock.py -c configs/ --config-name general_test.yaml \
    --account <pi_account>

# Full training run (resume from checkpoints if they exist)
python scripts/submit_sherlock.py -c configs/ --account <pi_account> --resume

# Single dataset
python scripts/submit_sherlock.py -c configs/slideseq/general.yaml --account <pi_account>
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `-c/--config` | required | `general.yaml`, per-model YAML, or directory |
| `--config-name` | `general.yaml` | Filename to search recursively in dirs |
| `--account` | `None` | Sherlock SLURM account/PI group |
| `--partition` | `gpu` | SLURM partition |
| `--resume` | `False` | Pass `--resume` to each `run all` invocation |
| `--dry-run` | `False` | Print job scripts without submitting |
| `--conda-prefix` | `$HOME/miniconda3` | Path to conda installation |
| `--env` | `factorization` | Conda environment name |
| `--skip-preprocess` | `False` | Skip the local preprocessing step |

### Resource table (first match wins on config path)

| Pattern | Time | Mem | CPUs/GPU |
|---------|------|-----|----------|
| `colon` | 72:00:00 | 128G | 8 |
| `liver/diseased` | 48:00:00 | 128G | 8 |
| `liver` | 24:00:00 | 64G | 8 |
| `merfish` | 24:00:00 | 64G | 8 |
| `slideseq` | 24:00:00 | 64G | 8 |
| `sdmbench` | 12:00:00 | 32G | 4 |
| `tenxvisium` | 8:00:00 | 32G | 4 |
| `osmfish` | 4:00:00 | 16G | 4 |
| *(default)* | 24:00:00 | 64G | 8 |

All jobs request **1 GPU** (one model = train → analyze → figures sequentially).

### Generated job script (written to `{output_dir}/logs/sherlock_{model_name}.sh`)

```bash
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gpus=1
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --time={time_limit}
#SBATCH --output={abs_output_dir}/logs/slurm_{model_name}_%j.log
#SBATCH --error={abs_output_dir}/logs/slurm_{model_name}_%j.log
# --account={account}   (uncommented only if --account provided)

source {conda_prefix}/etc/profile.d/conda.sh
conda activate {env_name}

cd {repo_root}

spatial_factorization run all -c {abs_config_path} [--resume]
```

- All paths are absolute (relative paths break when SLURM cwd differs)
- `repo_root` resolved at submission time via `Path(__file__).parent.parent`
- Job name derived from config path, e.g. `sdmbench_151507_mggp_svgp`, `liver_healthy_lcgp`
- SLURM logs sit alongside model logs in `{output_dir}/logs/`

---

## Verification

```bash
# 1. Test query_gpus fix (if GPUs available locally)
conda run -n factorization python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from spatial_factorization.runner import query_gpus
print([g.device_id for g in query_gpus()])  # should be [0]
"

# 2. Dry run — verify ~95 jobs with correct names and resource specs
python scripts/submit_sherlock.py -c configs/ --config-name general_test.yaml --dry-run

# 3. On Sherlock: submit one dataset as a smoke test
python scripts/submit_sherlock.py -c configs/slideseq/general_test.yaml \
    --account <pi_account>
squeue -u $USER   # verify 5 jobs appear
```
