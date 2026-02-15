"""Live status display for parallel training jobs."""

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table


@dataclass
class JobStatus:
    """Per-job state for status tracking."""

    name: str
    config_path: Path
    device: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    log_file: Optional[Path] = None
    return_code: Optional[int] = None
    epoch: int = 0
    total_epochs: int = 0
    elbo: float = 0.0
    remaining_time: str = ""  # e.g., "02:45" or "1:23:45"

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def elapsed_str(self) -> str:
        """Human-readable elapsed time."""
        return str(timedelta(seconds=int(self.elapsed)))


class StatusManager:
    """Coordinates live status display for parallel jobs."""

    def __init__(self, console: Optional[Console] = None):
        self.jobs: Dict[str, JobStatus] = {}
        self.console = console or Console()
        self._live: Optional[Live] = None
        self._lock = threading.Lock()

    def add_job(self, job: JobStatus) -> None:
        """Register a new job."""
        with self._lock:
            self.jobs[job.name] = job

    def update_job(self, name: str, **kwargs) -> None:
        """Update job fields (status, epoch, elbo, etc.)."""
        with self._lock:
            if name in self.jobs:
                for key, value in kwargs.items():
                    if hasattr(self.jobs[name], key):
                        setattr(self.jobs[name], key, value)

    def _make_table(self) -> Table:
        """Build the status table."""
        table = Table(title="Training Progress")
        table.add_column("Model", style="cyan", width=12)
        table.add_column("Device", width=8)
        table.add_column("Status", width=10)
        table.add_column("Epoch", width=12)
        table.add_column("ELBO", width=12)
        table.add_column("Remaining", width=10)
        table.add_column("Elapsed", width=10)

        status_styles = {
            "pending": "dim",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }

        for job in self.jobs.values():
            style = status_styles.get(job.status, "")

            epoch_str = f"{job.epoch}/{job.total_epochs}" if job.total_epochs else "-"
            elbo_str = f"{job.elbo:.1f}" if job.elbo else "-"
            remaining_str = job.remaining_time if job.remaining_time else "-"
            status_str = f"[{style}]{job.status}[/{style}]" if style else job.status

            table.add_row(
                job.name,
                job.device,
                status_str,
                epoch_str,
                elbo_str,
                remaining_str,
                job.elapsed_str(),
            )

        return table

    def start_live(self) -> None:
        """Start the live-updating display."""
        self._live = Live(
            self._make_table(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def refresh(self) -> None:
        """Refresh the display with current state."""
        if self._live:
            self._live.update(self._make_table())

    def stop_live(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def print_summary(self) -> None:
        """Print final summary table (after live display stopped)."""
        self.console.print("\n[bold]Training Complete[/bold]\n")
        self.console.print(self._make_table())

        self.console.print("\n[bold]Log Files:[/bold]")
        for job in self.jobs.values():
            if job.log_file:
                self.console.print(f"  {job.name}: {job.log_file}")

        # Summary stats
        completed = sum(1 for j in self.jobs.values() if j.status == "completed")
        failed = sum(1 for j in self.jobs.values() if j.status == "failed")
        total = len(self.jobs)

        if failed == 0:
            self.console.print(f"\n[green]All {total} models completed successfully.[/green]")
        else:
            self.console.print(f"\n[yellow]{completed}/{total} models completed, {failed} failed.[/yellow]")

    def __enter__(self) -> "StatusManager":
        self.start_live()
        return self

    def __exit__(self, *args) -> None:
        self.stop_live()


def _is_progress_line(line: str) -> bool:
    """Check if a line is a tqdm-style progress bar update.

    Progress lines typically look like:
    - "5000/10000 [01:23<02:45, 60.12it/s, ELBO=-5.475e+05]"
    - "transform_W:  38%|███▊      | 385/1000 [00:03<00:05, 109.52it/s, NLL=8353034.500000]"
    - " 10%|█         | 100/1000 [00:01<00:09, 100.00it/s]"
    """
    # Match tqdm patterns: N/M with bracket or pipe progress, or XX%|
    if re.search(r"\d+/\d+\s*[\[\|]", line):
        return True
    # Match percentage-based progress (with optional prefix like "transform_W:")
    if re.search(r"\d+%\|", line):
        return True
    return False


def stream_output(
    proc: subprocess.Popen,
    job_name: str,
    manager: StatusManager,
    log_file: Path,
) -> None:
    """
    Thread target: reads subprocess stdout with non-blocking I/O.

    - Writes non-progress lines to log file immediately
    - Buffers progress lines and writes only final state
    - Parses epoch/elbo from output
    - Updates StatusManager
    - Signals completion when stdout ends
    """
    import os
    import fcntl

    # Set stdout to non-blocking
    fd = proc.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    with open(log_file, "w") as log_fh:
        buffer = b""
        current_progress_line = ""  # Track current progress bar state

        while True:
            # Check if process has ended
            if proc.poll() is not None:
                # Read any remaining data
                try:
                    remaining = proc.stdout.read()
                    if remaining:
                        buffer += remaining
                except (BlockingIOError, IOError):
                    pass
                break

            try:
                chunk = proc.stdout.read(4096)
                if chunk:
                    buffer += chunk
            except (BlockingIOError, IOError):
                # No data available yet
                pass

            # Process complete lines (ending with \n)
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                decoded = line.decode(errors="replace").rstrip()

                # tqdm uses carriage returns, get the latest content
                if "\r" in decoded:
                    decoded = decoded.split("\r")[-1]

                # Skip empty lines
                if not decoded or decoded.isspace():
                    continue

                # Check if this is a progress line
                if _is_progress_line(decoded):
                    # Update current progress state (don't write yet)
                    current_progress_line = decoded
                else:
                    # Non-progress line: write any buffered progress first, then this line
                    if current_progress_line:
                        log_fh.write(current_progress_line + "\n")
                        log_fh.flush()
                        current_progress_line = ""
                    log_fh.write(decoded + "\n")
                    log_fh.flush()

                # Parse iteration/ELBO from output (for status updates)
                _parse_and_update(decoded, job_name, manager)

            # Check incomplete buffer for tqdm progress (ends with \r, no \n)
            if buffer and b"\r" in buffer:
                decoded = buffer.decode(errors="replace")
                latest = decoded.split("\r")[-1]
                if _is_progress_line(latest):
                    current_progress_line = latest
                _parse_and_update(latest, job_name, manager)

            # Small sleep to avoid busy-waiting
            time.sleep(0.1)

        # Write any final progress line
        if current_progress_line:
            log_fh.write(current_progress_line + "\n")
            log_fh.flush()

        # Process any remaining buffer
        if buffer:
            decoded = buffer.decode(errors="replace").rstrip()
            if "\r" in decoded:
                decoded = decoded.split("\r")[-1]
            if decoded and not decoded.isspace() and not _is_progress_line(decoded):
                log_fh.write(decoded + "\n")
                log_fh.flush()

    # Wait for process to fully complete
    proc.wait()
    manager.update_job(
        job_name,
        status="completed" if proc.returncode == 0 else "failed",
        end_time=time.time(),
        return_code=proc.returncode,
    )


def _parse_and_update(decoded: str, job_name: str, manager: StatusManager) -> None:
    """Parse iteration/ELBO from output line and update status."""
    # Formats:
    #   - verbose=True:  "Iteration 500: ELBO = -12345.67"
    #   - tqdm (stderr): "5000/10000 [01:23<02:45, 60.12it/s, ELBO=-5.475e+05, lr=5.2e-03]"

    iter_match = re.search(r"[Ii]teration[:\s]+(\d+)", decoded)
    progress_match = re.search(r"(\d+)/(\d+)\s*\[", decoded)  # tqdm: 5000/10000 [
    elbo_match = re.search(r"ELBO\s*[=:]\s*(-?[\d.eE+-]+)", decoded)  # handle scientific notation
    remaining_match = re.search(r"<(\d+:\d+(?::\d+)?)", decoded)  # tqdm: <02:45 or <1:23:45

    if iter_match:
        manager.update_job(job_name, epoch=int(iter_match.group(1)))
    if progress_match:
        manager.update_job(
            job_name,
            epoch=int(progress_match.group(1)),
            total_epochs=int(progress_match.group(2)),
        )
    if elbo_match:
        try:
            manager.update_job(job_name, elbo=float(elbo_match.group(1)))
        except ValueError:
            pass  # Invalid float format, skip
    if remaining_match:
        manager.update_job(job_name, remaining_time=remaining_match.group(1))
