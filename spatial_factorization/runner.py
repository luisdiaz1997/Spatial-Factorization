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
        force_preprocess: bool = False,
        dry_run: bool = False,
    ):
        """Initialize the job runner.

        Args:
            config_path: Path to general.yaml or per-model config.
            force_preprocess: Force re-run preprocessing even if exists.
            dry_run: Show plan without executing.
        """
        self.config_path = Path(config_path)
        self.force_preprocess = force_preprocess
        self.dry_run = dry_run
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

        # Phase 2: Preprocess once
        print("\nPhase 2: Checking preprocessing...")
        self._ensure_preprocessed(config_paths[0])

        # Set up log directory
        first_config = Config.from_yaml(config_paths[0])
        self.log_dir = Path(first_config.output_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3: Create jobs
        print("\nPhase 3: Creating jobs...")
        for path in config_paths:
            config = Config.from_yaml(path)
            log_file = self.log_dir / f"{config.model_name}.log"
            job = Job(name=config.model_name, config_path=path, log_file=log_file)
            self.jobs.append(job)
            self.run_status.jobs[job.name] = {
                "config_path": str(path),
                "status": "pending",
            }

            # Add train task to status manager
            self.status_manager.add_job(JobStatus(
                name=f"{config.model_name}_train",
                model=config.model_name,
                task="train",
                config_path=path,
                device="pending",
                status="pending",
                log_file=log_file,
                total_epochs=config.training.get("max_iter", 0),
            ))

            # Add analyze task to status manager
            self.status_manager.add_job(JobStatus(
                name=f"{config.model_name}_analyze",
                model=config.model_name,
                task="analyze",
                config_path=path,
                device="pending",
                status="pending",
                log_file=log_file,
                total_epochs=0,  # analyze doesn't have epochs in the same way
            ))

        self.is_multiplex = len(self.jobs) > 1

        self._print_job_summary()

        if self.dry_run:
            print("\n[Dry run] Would execute the above jobs. Exiting.")
            return

        # Phase 4: Parallel training + sequential analyze
        print("\nPhase 4: Parallel training + analyze (GPU + CPU fallback)...")
        self._run_parallel()

        # Phase 5: Report
        self._print_final_report()
        self._save_status()

    def _resolve_configs(self) -> List[Path]:
        """Resolve config paths to a list of per-model configs."""
        if self.config_path.is_dir():
            general_configs = list(self.config_path.rglob("general.yaml"))
            if not general_configs:
                raise ValueError(f"No general.yaml found in {self.config_path}")

            print(f"Found {len(general_configs)} general config(s)")

            config_paths = []
            for general_path in general_configs:
                from .generate import generate_configs

                generated = generate_configs(general_path)
                config_paths.extend(generated.values())
                print(f"  {general_path.parent.name}: {len(generated)} models")

            return config_paths

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
        print(f"\n{'Model':<15} {'Status':<10}")
        print("-" * 25)
        for job in self.jobs:
            print(f"{job.name:<15} {job.status:<10}")

    def _run_parallel(self) -> None:
        """Run training and analyze in parallel with GPU/CPU scheduling.

        Scheduling:
        - Training: Uses available GPUs + 1 CPU fallback
        - Analyze: Same pattern - parallel with GPU + CPU fallback
        - At least one job runs on CPU when all GPUs are busy
        """
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
                # === Start training jobs ===
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
                        elif not get_cpu_in_use():
                            # CPU fallback when all GPUs busy
                            job.device = "cpu"
                            train_cpu_slot = True
                            print(f"  [{job.name}] Training on CPU (GPUs busy)")
                        else:
                            continue
                    else:
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
                pending_train_jobs = [j for j in self.jobs if j.status == "pending"]
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
                            elif not get_cpu_in_use():
                                # CPU fallback when all GPUs busy
                                analyze_device = "cpu"
                                analyze_uses_cpu = True
                                analyze_cpu_slot = True
                                print(f"  [{job.name}] Analyzing on CPU (GPUs busy)")
                            else:
                                continue  # No resources available
                        else:
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

                        # Launch analyze+figures
                        env = self._get_launch_env(analyze_gpu_id)
                        proc = self._launch_process(["analyze", "figures"], job, env, task="analyze")

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
        cmd = [sys.executable, "-m", "spatial_factorization", "run"] + stages + ["-c", str(job.config_path)]

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

        config = Config.from_yaml(self.jobs[0].config_path)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        status_path = output_dir / "run_status.json"
        with open(status_path, "w") as f:
            json.dump(self.run_status.to_dict(), f, indent=2)

        print(f"\nRun status saved to: {status_path}")
