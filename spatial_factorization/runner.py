"""Job runner for parallel, resource-aware model training.

Provides GPU query/assignment and subprocess orchestration for training
multiple models in parallel with CPU/GPU resource management.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Config
from .status import JobStatus, StatusManager, stream_output


# Resource limits
CPU_SLOTS_PER_PROCESS = 16  # cores per training process
MAX_CPU_SLOTS = 64  # total cores available
# GPU scheduling: 1 job per GPU (exclusive access)


@dataclass
class GPUInfo:
    """Information about a GPU."""

    device_id: int
    name: str
    total_memory_gb: float
    free_memory_gb: float


def query_gpus() -> List[GPUInfo]:
    """Query all available GPUs via nvidia-smi (no CUDA context created).

    Returns:
        List of GPUInfo objects, one per GPU. Empty list if no GPUs available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            parts = [p.strip() for p in line.split(",")]
            gpus.append(GPUInfo(
                device_id=int(parts[0]),
                name=parts[1],
                total_memory_gb=float(parts[2]) / 1024,  # MiB → GiB
                free_memory_gb=float(parts[3]) / 1024,
            ))
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse GPU info: {line!r}: {e}", file=sys.stderr)

    return gpus


@dataclass
class Job:
    """A training job for a single model."""

    name: str
    config_path: Path
    stages: List[str] = field(default_factory=lambda: ["train", "analyze", "figures"])
    status: str = "pending"  # pending, training, analyzing, completed, failed
    device: str = "cpu"
    gpu_id: Optional[int] = None  # GPU used for training
    analyze_gpu_id: Optional[int] = None  # GPU used for analyze (may differ)
    analyze_uses_cpu: bool = False  # Whether analyze is using CPU
    pid: Optional[int] = None
    start_time: float = 0.0
    elapsed: float = 0.0
    error: Optional[str] = None
    log_file: Optional[Path] = None
    thread: Optional[threading.Thread] = None
    resume: bool = False  # Whether to resume training from a checkpoint
    video: bool = False   # Whether to capture training animation frames


@dataclass
class RunStatus:
    """Status of a multiplex run."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else (datetime.now() - self.start_time).total_seconds()
            ),
            "jobs": self.jobs,
        }


class JobRunner:
    """Parallel job runner with resource-aware scheduling.

    Scheduling:
    - Training: Uses available GPUs + 1 CPU fallback
    - Analyze: Same as training - parallel with GPU + CPU fallback
    - At least one job runs on CPU when all GPUs are busy
    """

    def __init__(
        self,
        config_path: str | Path,
        stages: List[str] = None,
        force_preprocess: bool = False,
        dry_run: bool = False,
        resume: bool = False,
        config_name: str = "general.yaml",
        failed_only: bool = False,
        video: bool = False,
        gpu_only: bool = False,
        no_heatmap: bool = False,
        skip_general: bool = False,
        probabilistic: bool = False,
        posterior_k: int | None = None,
        posterior_mem_gb: float | None = None,
        groups_derived: bool = False,
    ):
        """Initialize the job runner.

        Args:
            config_path: Path to general.yaml or per-model config.
            stages: Ordered list of pipeline stages to run. Defaults to full pipeline.
            force_preprocess: Force re-run preprocessing even if exists.
            dry_run: Show plan without executing.
            resume: Resume models with a checkpoint; train new ones from scratch.
            config_name: Filename to search for when config_path is a directory.
            failed_only: Only re-run jobs that failed in the previous run_status.json.
            gpu_only: Only assign jobs to GPUs; never fall back to CPU.
            no_heatmap: Skip celltype_gene_loadings and factor_gene_loadings heatmaps.
            skip_general: Skip general configs; treat all non-general yamls as per-model configs.
        """
        self.config_path = Path(config_path)
        self.stages = stages or ["preprocess", "train", "analyze", "figures"]
        self.force_preprocess = force_preprocess
        self.dry_run = dry_run
        self.resume = resume
        self.config_name = config_name
        self.failed_only = failed_only
        self.video = video
        self.gpu_only = gpu_only
        self.no_heatmap = no_heatmap
        self.skip_general = skip_general
        self.probabilistic = probabilistic
        self.posterior_k = posterior_k
        self.posterior_mem_gb = posterior_mem_gb
        self.groups_derived = groups_derived
        self.jobs: List[Job] = []
        self.run_status = RunStatus()
        self.status_manager = StatusManager()
        self.is_multiplex = False

    def run(self) -> None:
        """Execute the multiplex pipeline."""
        print(f"\n{'='*60}")
        print(f"  Multiplex Pipeline Runner")
        print(f"{'='*60}\n")

        # Phase 1: Detect config type
        print("Phase 1: Detecting config type...")
        config_paths = self._resolve_configs()

        if not config_paths:
            print("\nNothing to run.")
            return

        # Phase 2: Preprocess once per unique dataset (output_dir)
        if "preprocess" in self.stages:
            print("\nPhase 2: Checking preprocessing...")
            seen_output_dirs: set = set()
            for path in config_paths:
                cfg = Config.from_yaml(path)
                if cfg.output_dir not in seen_output_dirs:
                    self._ensure_preprocessed(path)
                    seen_output_dirs.add(cfg.output_dir)
        else:
            print("\nPhase 2: Skipping preprocessing (not in stages).")

        # Phase 3: Create jobs
        print("\nPhase 3: Creating jobs...")
        for path in config_paths:
            config = Config.from_yaml(path)

            # Per-job log directory (unique per dataset)
            log_dir = Path(config.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{config.model_name}.log"

            # Derive a unique job name from the output directory path so that
            # datasets with the same name but different paths (e.g. liver/healthy
            # vs liver/diseased) don't collide in the status dict.
            # e.g. "outputs/liver_test/healthy" → "liver_test_healthy_mggp_svgp"
            out_path = Path(config.output_dir)
            unique_prefix = "_".join(out_path.parts[1:]) if len(out_path.parts) > 1 else out_path.name
            job_name = f"{unique_prefix}_{config.model_name}"

            job = Job(name=job_name, config_path=path, log_file=log_file, resume=self.resume, video=self.video)
            self.jobs.append(job)
            self.run_status.jobs[job.name] = {
                "config_path": str(path),
                "status": "pending",
            }

            # Add train task to status manager (only if training is requested)
            if "train" in self.stages:
                self.status_manager.add_job(JobStatus(
                    name=f"{job_name}_train",
                    model=job_name,
                    task="train",
                    config_path=path,
                    device="pending",
                    status="pending",
                    log_file=log_file,
                    total_epochs=config.training.get("max_iter", 0),
                ))

            # Add analyze task to status manager (only if non-train stages are requested)
            non_train_stages = [s for s in self.stages if s not in ("train", "preprocess")]
            if non_train_stages:
                self.status_manager.add_job(JobStatus(
                    name=f"{job_name}_analyze",
                    model=job_name,
                    task="analyze",
                    config_path=path,
                    device="pending",
                    status="pending",
                    log_file=log_file,
                    total_epochs=0,
                ))

        self.is_multiplex = len(self.jobs) > 1

        self._print_job_summary()

        if self.dry_run:
            print("\n[Dry run] Would execute the above jobs. Exiting.")
            return

        # Phase 4: Parallel execution
        has_train = "train" in self.stages
        non_train_stages = [s for s in self.stages if s not in ("train", "preprocess")]
        if has_train:
            print("\nPhase 4: Parallel training + analyze (GPU + CPU fallback)...")
        elif non_train_stages:
            print(f"\nPhase 4: Parallel {' + '.join(non_train_stages)} (GPU + CPU fallback)...")
        else:
            print("\nPhase 4: Nothing to run in parallel.")
            return
        self._run_parallel()

        # Phase 5: Report
        self._print_final_report()
        self._save_status()

    def _get_status_path(self) -> Path:
        """Return the expected path to run_status.json for this run."""
        if self.config_path.is_dir():
            return self.config_path / "run_status.json"
        # For general.yaml or per-model config, look in the dataset output_dir.
        # Generate configs to find the first output_dir.
        if Config.is_general_config(self.config_path):
            from .generate import generate_configs
            generated = generate_configs(self.config_path)
            first_path = next(iter(generated.values()))
        else:
            first_path = self.config_path
        cfg = Config.from_yaml(first_path)
        return Path(cfg.output_dir) / "run_status.json"

    def _resolve_configs(self) -> List[Path]:
        """Resolve config paths to a list of per-model configs."""
        if self.failed_only:
            status_path = self._get_status_path()
            if not status_path.exists():
                raise ValueError(
                    f"No run_status.json found at {status_path}. "
                    "Run without --failed first to generate a status file."
                )
            with open(status_path) as f:
                prev_status = json.load(f)
            failed_configs = [
                Path(v["config_path"])
                for v in prev_status["jobs"].values()
                if v["status"] == "failed"
            ]
            if not failed_configs:
                print("No failed jobs found in run_status.json. Nothing to re-run.")
                return []
            print(f"Found {len(failed_configs)} failed job(s) to re-run:")
            for p in failed_configs:
                print(f"  {p}")
            return failed_configs

        if self.config_path.is_dir():
            if self.skip_general:
                # Skip general configs entirely; treat all non-general yamls as per-model
                per_model = sorted(
                    p for p in self.config_path.rglob("*.yaml")
                    if not Config.is_general_config(p)
                )
                if not per_model:
                    raise ValueError(f"No per-model yaml configs found in {self.config_path}")
                print(f"Found {len(per_model)} per-model config(s) (--skip-general)")
                for p in per_model:
                    print(f"  {p.name}")
                return per_model

            matched = list(self.config_path.rglob(self.config_name))

            if matched:
                # Split into general (needs expansion) vs per-model (use as-is)
                general_configs = [p for p in matched if Config.is_general_config(p)]
                per_model_configs = [p for p in matched if not Config.is_general_config(p)]

                config_paths = []
                if general_configs:
                    print(f"Found {len(general_configs)} general config(s)")
                    for general_path in general_configs:
                        from .generate import generate_configs
                        generated = generate_configs(general_path)
                        config_paths.extend(generated.values())
                        print(f"  {general_path.parent.name}: {len(generated)} models")
                if per_model_configs:
                    print(f"Found {len(per_model_configs)} per-model config(s) matching {self.config_name}")
                    for p in per_model_configs:
                        print(f"  {p.parent.name}/{p.name}")
                    config_paths.extend(per_model_configs)
                return config_paths

            # No general configs found — treat every yaml in the directory as a
            # per-model config (e.g. configs/slideseq/video/ with named variants)
            per_model = sorted(self.config_path.rglob("*.yaml"))
            if not per_model:
                raise ValueError(f"No yaml configs found in {self.config_path}")
            print(f"Found {len(per_model)} per-model config(s) (no {self.config_name})")
            for p in per_model:
                print(f"  {p.name}")
            return per_model

        if Config.is_general_config(self.config_path):
            from .generate import generate_configs

            print(f"General config detected: {self.config_path}")
            generated = generate_configs(self.config_path)
            return list(generated.values())

        print(f"Per-model config: {self.config_path}")
        return [self.config_path]

    def _ensure_preprocessed(self, config_path: Path) -> None:
        """Ensure preprocessed data exists, run if needed."""
        config = Config.from_yaml(config_path)
        output_dir = Path(config.output_dir)

        if Config.preprocessed_exists(output_dir) and not self.force_preprocess:
            print(f"  Preprocessed data exists: {output_dir}/preprocessed/")
            return

        if self.force_preprocess:
            print(f"  Force re-processing requested...")

        print(f"  Running preprocessing...")
        self._run_stage("preprocess", config_path)

    def _print_job_summary(self) -> None:
        """Print a table of jobs to be run."""
        col_w = max(15, max(len(j.name) for j in self.jobs) + 2) if self.jobs else 15
        print(f"\n{'Model':<{col_w}} {'Status':<10}")
        print("-" * (col_w + 12))
        for job in self.jobs:
            print(f"{job.name:<{col_w}} {job.status:<10}")

    def _run_parallel(self) -> None:
        """Run training and analyze in parallel with GPU/CPU scheduling.

        Scheduling:
        - Training: Uses available GPUs + 1 CPU fallback
        - Analyze: Same pattern - parallel with GPU + CPU fallback
        - At least one job runs on CPU when all GPUs are busy

        Mode A (train in self.stages): train first, then analyze/figures after each job completes.
        Mode B (train not in self.stages): skip training, all jobs go directly to analyze queue.
        """
        non_train_stages = [s for s in self.stages if s not in ("train", "preprocess")]
        mode_b = "train" not in self.stages

        gpus = query_gpus()
        if gpus:
            print(f"Available GPUs:")
            for gpu in gpus:
                print(f"  GPU {gpu.device_id}: {gpu.name} ({gpu.free_memory_gb:.1f}GB free)")
        else:
            print("No GPUs available, all jobs will run on CPU")

        # Training state
        training_running: Dict[int, Tuple[subprocess.Popen, Job]] = {}
        training_done: Set[str] = set()  # Jobs that finished training (success or fail)
        train_gpu_slots: Set[int] = set()  # GPU IDs in use by training
        train_cpu_slot = False  # CPU slot in use by training?

        # Analyze state
        analyze_pending: Set[str] = set()  # Jobs waiting for training to finish before analyze
        analyze_running: Dict[int, Tuple[subprocess.Popen, Job]] = {}  # pid -> (proc, job)
        analyze_gpu_slots: Set[int] = set()  # GPU IDs in use by analyze
        analyze_cpu_slot = False  # CPU slot in use by analyze?
        analyze_done: Set[str] = set()

        total = len(self.jobs)

        # Mode B: skip training, pre-populate queues so all jobs go directly to analyze
        if mode_b:
            training_done = {job.name for job in self.jobs}
            analyze_pending = {job.name for job in self.jobs}

        def signal_handler(sig, frame):
            self.status_manager.stop_live()
            print("\n\nInterrupted! Terminating...")
            for proc, _ in training_running.values():
                proc.terminate()
            for proc, _ in analyze_running.values():
                proc.terminate()
            self._save_status()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        def get_used_gpus() -> Set[int]:
            """Get all GPU IDs currently in use by training or analyze."""
            return train_gpu_slots | analyze_gpu_slots

        def get_cpu_in_use() -> bool:
            """Check if CPU slot is in use by either training or analyze."""
            return train_cpu_slot or analyze_cpu_slot

        with self.status_manager:
            while len(analyze_done) < total:
                # === Start training jobs (Mode A only) ===
                if not mode_b:
                    for job in self.jobs:
                        if job.status != "pending":
                            continue

                        # Check if we can start this job
                        config = Config.from_yaml(job.config_path)
                        use_gpu = config.training.get("device", "cpu") == "gpu" and gpus

                        gpu_id = None
                        if use_gpu:
                            # Find free GPU (not used by training OR analyze)
                            used_gpus = get_used_gpus()
                            free = [g.device_id for g in gpus if g.device_id not in used_gpus]
                            if free:
                                gpu_id = free[0]
                                job.device = f"cuda:{gpu_id}"
                                job.gpu_id = gpu_id
                                train_gpu_slots.add(gpu_id)
                                print(f"  [{job.name}] Training on GPU {gpu_id}")
                            elif self.gpu_only:
                                continue  # Wait for a GPU to free up
                            elif not get_cpu_in_use():
                                # CPU fallback when all GPUs busy
                                job.device = "cpu"
                                train_cpu_slot = True
                                print(f"  [{job.name}] Training on CPU (GPUs busy)")
                            else:
                                continue
                        else:
                            if self.gpu_only:
                                continue  # Never assign CPU when gpu_only
                            if not get_cpu_in_use():
                                job.device = "cpu"
                                train_cpu_slot = True
                                print(f"  [{job.name}] Training on CPU")
                            else:
                                continue

                        # Launch training
                        self.status_manager.update_job(f"{job.name}_train", device=job.device)
                        env = self._get_launch_env(job.gpu_id)
                        proc = self._launch_process(["train"], job, env, task="train")

                        if proc:
                            job.status = "training"
                            job.start_time = time.time()
                            job.pid = proc.pid
                            training_running[job.pid] = (proc, job)
                            analyze_pending.add(job.name)  # Queue for analyze after training
                            self.status_manager.update_job(f"{job.name}_train", status="training", start_time=time.time())
                        else:
                            job.status = "failed"
                            job.error = "Failed to launch training"
                            training_done.add(job.name)
                            analyze_done.add(job.name)
                            self.run_status.jobs[job.name]["status"] = "failed"
                            self.run_status.jobs[job.name]["error"] = job.error
                            self.status_manager.update_job(f"{job.name}_train", status="failed")
                            self.status_manager.update_job(f"{job.name}_analyze", status="skipped")

                    # === Check completed training ===
                    finished = [pid for pid, (proc, _) in training_running.items() if proc.poll() is not None]

                    for pid in finished:
                        proc, job = training_running.pop(pid)

                        # Free device
                        if job.gpu_id is not None:
                            train_gpu_slots.discard(job.gpu_id)
                            print(f"  [{job.name}] Training done, GPU {job.gpu_id} freed")
                        else:
                            train_cpu_slot = False
                            print(f"  [{job.name}] Training done, CPU freed")

                        job.elapsed = time.time() - job.start_time
                        training_done.add(job.name)

                        if proc.returncode != 0:
                            job.status = "failed"
                            job.error = f"Training exit code: {proc.returncode}"
                            analyze_pending.discard(job.name)
                            analyze_done.add(job.name)
                            self.run_status.jobs[job.name]["status"] = "failed"
                            self.run_status.jobs[job.name]["error"] = job.error
                            self.status_manager.update_job(f"{job.name}_train", status="failed")
                            self.status_manager.update_job(f"{job.name}_analyze", status="skipped")
                        else:
                            # Training completed successfully
                            self.status_manager.update_job(f"{job.name}_train", status="completed", end_time=time.time())

                # === Start analyze jobs (parallel with GPU/CPU scheduling) ===
                # Priority: training jobs get first dibs on GPUs
                # Only start analyze if no pending training jobs are waiting
                # In Mode B, pending_train_jobs is always empty (no training phase)
                pending_train_jobs = [] if mode_b else [j for j in self.jobs if j.status == "pending"]
                has_free_gpu = any(g.device_id not in get_used_gpus() for g in gpus)
                has_free_cpu = not get_cpu_in_use()

                # If there are pending training jobs and resources available, skip analyze
                if pending_train_jobs and (has_free_gpu or has_free_cpu):
                    pass  # Let training jobs get resources first
                else:
                    for job in self.jobs:
                        # Skip if not ready for analyze
                        if job.name not in analyze_pending:
                            continue
                        if job.name not in training_done:
                            continue
                        if job.name in analyze_done:
                            continue
                        if job.name in {j.name for _, j in analyze_running.values()}:
                            continue

                        # Skip failed jobs
                        if job.status == "failed":
                            analyze_pending.discard(job.name)
                            analyze_done.add(job.name)
                            continue

                        # Determine device for analyze
                        config = Config.from_yaml(job.config_path)
                        use_gpu = config.training.get("device", "cpu") == "gpu" and gpus

                        analyze_gpu_id = None
                        analyze_uses_cpu = False
                        if use_gpu:
                            # Find free GPU (not used by training OR analyze)
                            used_gpus = get_used_gpus()
                            free = [g.device_id for g in gpus if g.device_id not in used_gpus]
                            if free:
                                analyze_gpu_id = free[0]
                                analyze_device = f"cuda:{analyze_gpu_id}"
                                analyze_gpu_slots.add(analyze_gpu_id)
                                print(f"  [{job.name}] Analyzing on GPU {analyze_gpu_id}")
                            elif self.gpu_only:
                                continue  # Wait for a GPU to free up
                            elif not get_cpu_in_use():
                                # CPU fallback when all GPUs busy
                                analyze_device = "cpu"
                                analyze_uses_cpu = True
                                analyze_cpu_slot = True
                                print(f"  [{job.name}] Analyzing on CPU (GPUs busy)")
                            else:
                                continue  # No resources available
                        else:
                            if self.gpu_only:
                                continue  # Never assign CPU when gpu_only
                            if not get_cpu_in_use():
                                analyze_device = "cpu"
                                analyze_uses_cpu = True
                                analyze_cpu_slot = True
                                print(f"  [{job.name}] Analyzing on CPU")
                            else:
                                continue

                        # Store analyze device info on job
                        job.analyze_gpu_id = analyze_gpu_id
                        job.analyze_uses_cpu = analyze_uses_cpu

                        # Remove from pending
                        analyze_pending.discard(job.name)

                        # Update status
                        self.status_manager.update_job(f"{job.name}_analyze", status="analyzing", device=analyze_device)

                        # Launch non-train stages (analyze, figures, or subset thereof)
                        env = self._get_launch_env(analyze_gpu_id)
                        proc = self._launch_process(non_train_stages, job, env, task="analyze")

                        if proc:
                            analyze_running[proc.pid] = (proc, job)
                        else:
                            job.status = "failed"
                            job.error = "Failed to launch analyze"
                            analyze_done.add(job.name)
                            self.run_status.jobs[job.name]["status"] = "failed"
                            self.run_status.jobs[job.name]["error"] = job.error
                            self.status_manager.update_job(f"{job.name}_analyze", status="failed")
                            # Free CPU slot if we took it
                            if analyze_uses_cpu:
                                analyze_cpu_slot = False

                # === Check completed analyze ===
                finished_analyze = [pid for pid, (proc, _) in analyze_running.items() if proc.poll() is not None]

                for pid in finished_analyze:
                    proc, job = analyze_running.pop(pid)

                    # Free device used by analyze
                    if job.analyze_gpu_id is not None:
                        analyze_gpu_slots.discard(job.analyze_gpu_id)
                        print(f"  [{job.name}] Analyze done, GPU {job.analyze_gpu_id} freed")
                    elif getattr(job, 'analyze_uses_cpu', False):
                        analyze_cpu_slot = False
                        print(f"  [{job.name}] Analyze done, CPU freed")

                    if proc.returncode == 0:
                        job.status = "completed"
                        self.run_status.jobs[job.name]["status"] = "completed"
                        self.run_status.jobs[job.name]["elapsed"] = job.elapsed
                        self.status_manager.update_job(f"{job.name}_analyze", status="completed", end_time=time.time())
                        print(f"  [{job.name}] Complete!")
                    else:
                        job.status = "failed"
                        job.error = f"Analyze exit code: {proc.returncode}"
                        self.run_status.jobs[job.name]["status"] = "failed"
                        self.run_status.jobs[job.name]["error"] = job.error
                        self.status_manager.update_job(f"{job.name}_analyze", status="failed")

                    analyze_done.add(job.name)

                self.status_manager.refresh()
                time.sleep(0.5)

    def _launch_process(self, stages: List[str], job: Job, env: Dict[str, str], task: str = "train") -> Optional[subprocess.Popen]:
        """Launch a subprocess for given stages."""
        cmd = [sys.executable, "-m", "spatial_factorization", "run"] + stages
        if task == "train" and job.resume:
            cmd.append("--resume")
        if task == "train" and job.video:
            cmd.append("--video")
        if "figures" in stages and self.no_heatmap:
            cmd.append("--no-heatmap")
        if self.probabilistic and (("analyze" in stages) or ("train" in stages)):
            cmd.append("--probabilistic")
        if self.posterior_k is not None and "analyze" in stages:
            cmd += ["--posterior-k", str(self.posterior_k)]
        if self.posterior_mem_gb is not None and "analyze" in stages:
            cmd += ["--posterior-mem-gb", str(self.posterior_mem_gb)]
        if self.groups_derived and "analyze" in stages:
            cmd.append("--groups-derived")
        cmd += ["-c", str(job.config_path)]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
            )

            # Stream output - use task-specific job name for status updates
            status_job_name = f"{job.name}_{task}"
            thread = threading.Thread(
                target=stream_output,
                args=(proc, status_job_name, self.status_manager, job.log_file),
                daemon=True,
            )
            thread.start()
            return proc
        except Exception as e:
            print(f"  [{job.name}] Failed to launch: {e}")
            return None

    def _get_launch_env(self, gpu_id: Optional[int]) -> Dict[str, str]:
        """Get environment variables for launching a subprocess."""
        env = {"PYTHONUNBUFFERED": "1"}
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def _run_stage(self, stage: str, config_path: Path, log_file: Optional[Path] = None, force_cpu: bool = False) -> subprocess.CompletedProcess:
        """Run a single pipeline stage as a subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "spatial_factorization",
            stage,
            "-c",
            str(config_path),
        ]
        if stage == "figures" and self.no_heatmap:
            cmd.append("--no-heatmap")
        if stage in ("analyze", "train") and self.probabilistic:
            cmd.append("--probabilistic")
        if stage == "analyze" and self.posterior_k is not None:
            cmd += ["--posterior-k", str(self.posterior_k)]
        if stage == "analyze" and self.posterior_mem_gb is not None:
            cmd += ["--posterior-mem-gb", str(self.posterior_mem_gb)]
        if stage == "analyze" and self.groups_derived:
            cmd.append("--groups-derived")

        env = dict(os.environ)
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if log_file:
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n  Stage: {stage}\n{'='*60}\n\n")
                if result.stdout:
                    f.write(result.stdout)
                if result.returncode != 0:
                    f.write(f"\n[FAILED] Exit code: {result.returncode}\n")

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        return result

    def _print_final_report(self) -> None:
        """Print final run report."""
        self.run_status.end_time = datetime.now()
        self.status_manager.print_summary()

        total_elapsed = (self.run_status.end_time - self.run_status.start_time).total_seconds()
        print(f"\nTotal wall time: {timedelta(seconds=int(total_elapsed))}")

    def _save_status(self) -> None:
        """Save run status to JSON file."""
        if not self.jobs:
            return

        # Any job still "pending" was never completed (interrupted or never started).
        # Mark it as "failed" so --failed picks it up on the next run.
        for job in self.jobs:
            if self.run_status.jobs[job.name]["status"] == "pending":
                self.run_status.jobs[job.name]["status"] = "failed"
                self.run_status.jobs[job.name]["error"] = "interrupted"

        # For directory-based runs, save alongside the config directory
        # For single-dataset runs, save in the dataset output_dir
        if self.config_path.is_dir():
            status_dir = self.config_path
        else:
            config = Config.from_yaml(self.jobs[0].config_path)
            status_dir = Path(config.output_dir)
            status_dir.mkdir(parents=True, exist_ok=True)

        status_path = status_dir / "run_status.json"

        # Merge with existing status so jobs from previous runs are preserved.
        # Only the jobs from the current run are updated.
        merged = self.run_status.to_dict()
        if status_path.exists():
            try:
                with open(status_path) as f:
                    prev = json.load(f)
                # Start from previous jobs, then overlay current run's results
                combined_jobs = prev.get("jobs", {})
                combined_jobs.update(merged["jobs"])
                merged["jobs"] = combined_jobs
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupt file — just write fresh

        with open(status_path, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"\nRun status saved to: {status_path}")
