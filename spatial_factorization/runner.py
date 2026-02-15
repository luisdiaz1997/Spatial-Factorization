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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .status import JobStatus, StatusManager, stream_output


# Resource limits
CPU_SLOTS_PER_PROCESS = 16  # cores per training process
MAX_CPU_SLOTS = 64  # total cores available
GPU_MEMORY_BUDGET_GB = 11  # GB reserved per job in our virtual tracking


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


def assign_gpu(gpus: List[GPUInfo], gpu_slots_used: Dict[int, int] = None) -> Optional[int]:
    """Assign a GPU using virtual free memory from our slot tracking.

    Uses total GPU memory minus reserved slots to make assignment decisions,
    NOT real-time free memory. This is necessary because training jobs don't
    consume GPU memory immediately during initialization (K-means, KNN, etc.),
    but must be considered "reserved" as soon as assigned.

    Args:
        gpus: List of available GPUInfo objects.
        gpu_slots_used: Dict mapping gpu_id -> number of running jobs on that GPU.

    Returns:
        GPU device ID with most virtual free memory (≥ budget).
        None if no GPU has sufficient virtual capacity.
    """
    if not gpus:
        return None

    gpu_slots_used = gpu_slots_used or {}

    best_gpu = None
    best_virtual_free = 0

    for gpu in gpus:
        slots = gpu_slots_used.get(gpu.device_id, 0)
        virtual_free_gb = gpu.total_memory_gb - (slots * GPU_MEMORY_BUDGET_GB)

        if virtual_free_gb >= GPU_MEMORY_BUDGET_GB:
            if virtual_free_gb > best_virtual_free:
                best_virtual_free = virtual_free_gb
                best_gpu = gpu.device_id

    return best_gpu


@dataclass
class Job:
    """A training job for a single model."""

    name: str
    config_path: Path
    stages: List[str] = field(default_factory=lambda: ["train", "analyze", "figures"])
    status: str = "pending"  # pending, running, completed, failed
    device: str = "cpu"
    gpu_id: Optional[int] = None
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

    Phases:
    1. Detect config type (general.yaml vs per-model)
    2. Preprocess once (if needed)
    3. Create jobs from per-model configs
    4. Parallel training with CPU/GPU management
    5. Streaming post-train (analyze → figures per model)
    6. Report results
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

            # Add to status manager
            self.status_manager.add_job(JobStatus(
                name=config.model_name,
                config_path=path,
                device="pending",
                status="pending",
                log_file=log_file,
                total_epochs=config.training.get("max_iter", 0),
            ))

        self._print_job_summary()

        if self.dry_run:
            print("\n[Dry run] Would execute the above jobs. Exiting.")
            return

        # Phase 4: Parallel training
        print("\nPhase 4: Parallel training...")
        self._run_parallel_training()

        # Phase 5: Streaming post-train (done within training loop)

        # Phase 6: Report
        self._print_final_report()
        self._save_status()

    def _resolve_configs(self) -> List[Path]:
        """Resolve config paths to a list of per-model configs.

        Returns:
            List of paths to per-model config files.
        """
        # If directory, find all general.yaml files
        if self.config_path.is_dir():
            general_configs = list(self.config_path.rglob("general.yaml"))
            if not general_configs:
                raise ValueError(f"No general.yaml found in {self.config_path}")

            print(f"Found {len(general_configs)} general config(s)")

            # Generate per-model configs for each general.yaml
            config_paths = []
            for general_path in general_configs:
                from .generate import generate_configs

                generated = generate_configs(general_path)
                config_paths.extend(generated.values())
                print(f"  {general_path.parent.name}: {len(generated)} models")

            return config_paths

        # If general.yaml, generate per-model configs
        if Config.is_general_config(self.config_path):
            from .generate import generate_configs

            print(f"General config detected: {self.config_path}")
            generated = generate_configs(self.config_path)
            return list(generated.values())

        # Single per-model config
        print(f"Per-model config: {self.config_path}")
        return [self.config_path]

    def _ensure_preprocessed(self, config_path: Path) -> None:
        """Ensure preprocessed data exists, run if needed."""
        config = Config.from_yaml(config_path)
        output_dir = Path(config.output_dir)  # Already includes dataset in path

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

    def _run_parallel_training(self) -> None:
        """Run training jobs in parallel with resource management."""
        # Calculate max concurrent jobs based on CPU
        max_concurrent = min(MAX_CPU_SLOTS // CPU_SLOTS_PER_PROCESS, len(self.jobs))

        # Query GPUs
        gpus = query_gpus()
        if gpus:
            print(f"Available GPUs:")
            for gpu in gpus:
                print(f"  GPU {gpu.device_id}: {gpu.name} ({gpu.free_memory_gb:.1f}GB free / {gpu.total_memory_gb:.1f}GB total)")
        else:
            print("No GPUs available, all jobs will run on CPU")

        # Track running jobs
        running: Dict[int, Tuple[subprocess.Popen, Job]] = {}
        cpu_slots_used = 0
        gpu_slots_used: Dict[int, int] = {}  # gpu_id -> count

        # Track completed jobs
        completed = 0
        failed_to_start = 0
        total = len(self.jobs)

        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.status_manager.stop_live()
            print("\n\nInterrupted! Terminating subprocesses...")
            for proc, job in running.values():
                proc.terminate()
            self._save_status()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        # Start live status display
        with self.status_manager:
            # Main event loop
            while completed < total:
                # Start new jobs if slots available
                pending_jobs = [j for j in self.jobs if j.status == "pending"]
                for job in pending_jobs:
                    if len(running) >= max_concurrent:
                        break

                    # Determine device
                    config = Config.from_yaml(job.config_path)
                    use_gpu = config.training.get("device", "cpu") == "gpu" and gpus

                    gpu_id = None
                    if use_gpu:
                        gpu_id = assign_gpu(gpus, gpu_slots_used)
                        if gpu_id is not None:
                            job.device = f"cuda:{gpu_id}"
                            job.gpu_id = gpu_id
                            gpu_slots_used[gpu_id] = gpu_slots_used.get(gpu_id, 0) + 1
                        else:
                            job.device = "cpu"
                    else:
                        job.device = "cpu"

                    # Update status manager with device
                    self.status_manager.update_job(job.name, device=job.device)

                    # Launch job
                    env = self._get_launch_env(job.gpu_id)
                    proc, thread = self._launch_training(job, env)

                    if proc:
                        job.status = "running"
                        job.start_time = time.time()
                        job.pid = proc.pid
                        job.thread = thread
                        running[job.pid] = (proc, job)
                        cpu_slots_used += CPU_SLOTS_PER_PROCESS

                        # Update status manager
                        self.status_manager.update_job(
                            job.name,
                            status="running",
                            start_time=time.time(),
                        )
                    else:
                        # Failed to start
                        job.status = "failed"
                        job.error = "Failed to launch subprocess"
                        failed_to_start += 1
                        completed += 1  # Count as completed so loop progresses
                        self.run_status.jobs[job.name]["status"] = "failed"
                        self.run_status.jobs[job.name]["error"] = job.error
                        self.status_manager.update_job(
                            job.name,
                            status="failed",
                            end_time=time.time(),
                        )

                # Check if we can make progress
                if not running:
                    still_pending = [j for j in self.jobs if j.status == "pending"]
                    if still_pending and len(running) < max_concurrent:
                        pass  # Will be shown in status table
                    elif completed >= total:
                        break
                    elif failed_to_start == total:
                        break

                # Check running jobs
                finished_pids = []
                for pid, (proc, job) in running.items():
                    if proc.poll() is not None:
                        finished_pids.append(pid)

                # Process completed jobs
                for pid in finished_pids:
                    proc, job = running.pop(pid)
                    cpu_slots_used -= CPU_SLOTS_PER_PROCESS

                    # Free GPU slot
                    if job.gpu_id is not None:
                        gpu_slots_used[job.gpu_id] -= 1
                        if gpu_slots_used[job.gpu_id] == 0:
                            del gpu_slots_used[job.gpu_id]

                    elapsed = time.time() - job.start_time
                    job.elapsed = elapsed

                    if proc.returncode == 0:
                        # Training completed, now run post-train stages
                        # Don't mark as completed until post-train succeeds
                        self.status_manager.update_job(
                            job.name,
                            status="analyzing",  # Show progress
                        )

                        post_train_error = self._run_post_train(job)

                        if post_train_error:
                            job.status = "failed"
                            job.error = post_train_error
                            self.run_status.jobs[job.name]["status"] = "failed"
                            self.run_status.jobs[job.name]["error"] = post_train_error
                            self.status_manager.update_job(
                                job.name,
                                status="failed",
                                end_time=time.time(),
                            )
                        else:
                            job.status = "completed"
                            self.run_status.jobs[job.name]["status"] = "completed"
                            self.run_status.jobs[job.name]["elapsed"] = elapsed
                            self.status_manager.update_job(
                                job.name,
                                status="completed",
                                end_time=time.time(),
                            )

                        completed += 1
                    else:
                        job.status = "failed"
                        job.error = f"Exit code: {proc.returncode}"
                        completed += 1
                        self.run_status.jobs[job.name]["status"] = "failed"
                        self.run_status.jobs[job.name]["error"] = job.error
                        self.status_manager.update_job(
                            job.name,
                            status="failed",
                            end_time=time.time(),
                        )

                # Refresh display
                self.status_manager.refresh()

                # Sleep before next poll
                time.sleep(0.5)

    def _run_post_train(self, job: Job) -> Optional[str]:
        """Run post-training stages (analyze, figures) for a job.

        Returns:
            Error message if any stage failed, None if all succeeded.
        """
        for stage in ["analyze", "figures"]:
            try:
                self._run_stage(stage, job.config_path, job.log_file)
            except Exception as e:
                return f"{stage}: {e}"
        return None

    def _launch_training(self, job: Job, env: Dict[str, str]) -> Optional[Tuple[subprocess.Popen, threading.Thread]]:
        """Launch a training job as a subprocess with output streaming.

        Args:
            job: The job to launch.
            env: Environment variables with CUDA device set.

        Returns:
            Tuple of (subprocess.Popen, threading.Thread), or (None, None) if launch failed.
        """
        cmd = [
            sys.executable,
            "-m",
            "spatial_factorization",
            "run",
            "train",
            "-c",
            str(job.config_path),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
            )

            # Start thread to stream output
            thread = threading.Thread(
                target=stream_output,
                args=(proc, job.name, self.status_manager, job.log_file),
                daemon=True,
            )
            thread.start()

            return proc, thread
        except Exception as e:
            print(f"  [{job.name}] Failed to launch: {e}")
            return None, None

    def _get_launch_env(self, gpu_id: Optional[int]) -> Dict[str, str]:
        """Get environment variables for launching a subprocess.

        Args:
            gpu_id: GPU device ID, or None for CPU.

        Returns:
            Dictionary of environment variables.
        """
        env = {"PYTHONUNBUFFERED": "1"}
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return env

    def _run_stage(self, stage: str, config_path: Path, log_file: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a single pipeline stage as a subprocess.

        Args:
            stage: Stage name (preprocess, train, analyze, figures).
            config_path: Path to config file.
            log_file: Optional path to log file for output.

        Returns:
            CompletedProcess result.
        """
        cmd = [
            sys.executable,
            "-m",
            "spatial_factorization",
            stage,
            "-c",
            str(config_path),
        ]

        # Run with output capture
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Log output if log_file provided
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"  Stage: {stage}\n")
                f.write(f"{'='*60}\n\n")
                if result.stdout:
                    f.write(result.stdout)
                if result.stderr:
                    f.write(f"\n[stderr]:\n{result.stderr}")
                f.write(f"\n{'='*60}\n\n")
                if result.returncode != 0:
                    f.write(f"[FAILED] Exit code: {result.returncode}\n")

        # Raise on failure
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        return result

    def _print_final_report(self) -> None:
        """Print final run report."""
        self.run_status.end_time = datetime.now()

        # Use status manager's summary
        self.status_manager.print_summary()

        total_elapsed = (self.run_status.end_time - self.run_status.start_time).total_seconds()
        print(f"\nTotal wall time: {timedelta(seconds=int(total_elapsed))}")

    def _save_status(self) -> None:
        """Save run status to JSON file."""
        # Use output dir from first job
        if not self.jobs:
            return

        config = Config.from_yaml(self.jobs[0].config_path)
        output_dir = Path(config.output_dir)  # Already includes dataset in path
        output_dir.mkdir(parents=True, exist_ok=True)

        status_path = output_dir / "run_status.json"
        with open(status_path, "w") as f:
            json.dump(self.run_status.to_dict(), f, indent=2)

        print(f"\nRun status saved to: {status_path}")
