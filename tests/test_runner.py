"""Tests for job runner with mocked GPUs."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from spatial_factorization.config import Config
from spatial_factorization.runner import (
    GPUInfo,
    Job,
    JobRunner,
    RunStatus,
    assign_gpu,
    query_gpus,
)


@pytest.fixture
def mock_gpu():
    """Create a mock GPU info."""
    return GPUInfo(
        device_id=0,
        name="NVIDIA A30",
        total_memory_gb=24.0,
        free_memory_gb=20.0,
    )


@pytest.fixture
def mock_gpu_low_memory():
    """Create a mock GPU with low memory."""
    return GPUInfo(
        device_id=1,
        name="NVIDIA A30",
        total_memory_gb=24.0,
        free_memory_gb=8.0,
    )


def test_gpu_info_from_device():
    """Test GPUInfo creation."""
    gpu = GPUInfo(device_id=0, name="A30", total_memory_gb=24.0, free_memory_gb=20.0)
    assert gpu.device_id == 0
    assert gpu.name == "A30"
    assert gpu.total_memory_gb == 24.0
    assert gpu.free_memory_gb == 20.0


def test_assign_gpu_with_sufficient_memory(mock_gpu, mock_gpu_low_memory):
    """Test assign_gpu selects GPU with capacity (uses total_memory for virtual tracking)."""
    gpus = [mock_gpu, mock_gpu_low_memory]
    result = assign_gpu(gpus, {})
    # Both have 24GB total with 0 slots, so both have 24GB virtual free
    # GPU 0 wins (first seen at equal virtual free)
    assert result == 0


def test_assign_gpu_all_at_capacity():
    """Test assign_gpu returns None when all GPUs at virtual capacity."""
    gpu = GPUInfo(device_id=0, name="A30", total_memory_gb=24.0, free_memory_gb=20.0)
    # 3 slots x 11GB budget = 33GB reserved > 24GB total → no capacity
    result = assign_gpu([gpu], {0: 3})
    assert result is None


def test_assign_gpu_small_gpu_below_budget():
    """Test assign_gpu skips GPUs too small for budget."""
    small_gpu = GPUInfo(device_id=0, name="Small", total_memory_gb=8.0, free_memory_gb=8.0)
    result = assign_gpu([small_gpu], {})
    # 8GB total < 11GB budget → not eligible
    assert result is None


def test_assign_gpu_empty_list():
    """Test assign_gpu with empty GPU list."""
    result = assign_gpu([], {})
    assert result is None


def test_assign_gpu_virtual_tracking_spreads_jobs():
    """Test assign_gpu spreads jobs across GPUs using virtual memory tracking."""
    gpu0 = GPUInfo(device_id=0, name="A30", total_memory_gb=24.0, free_memory_gb=24.0)
    gpu1 = GPUInfo(device_id=1, name="A30", total_memory_gb=24.0, free_memory_gb=24.0)

    gpus = [gpu0, gpu1]

    # Job 1: Both have 24GB virtual free → GPU 0 (first seen)
    result1 = assign_gpu(gpus, {})
    assert result1 == 0

    # Job 2: GPU 0 vfree=12, GPU 1 vfree=24 → GPU 1
    result2 = assign_gpu(gpus, {0: 1})
    assert result2 == 1

    # Job 3: GPU 0 vfree=12, GPU 1 vfree=12 → GPU 0 (first seen at tie)
    result3 = assign_gpu(gpus, {0: 1, 1: 1})
    assert result3 == 0

    # Job 4: GPU 0 vfree=0 (at capacity), GPU 1 vfree=12 → GPU 1
    result4 = assign_gpu(gpus, {0: 2, 1: 1})
    assert result4 == 1

    # All full: GPU 0 vfree=0, GPU 1 vfree=0 → None
    result5 = assign_gpu(gpus, {0: 2, 1: 2})
    assert result5 is None


def test_assign_gpu_ignores_actual_free_memory():
    """Test that assign_gpu uses total memory, not actual free memory.

    This is the core fix: even if GPUs have slightly different actual free
    memory (e.g., 23.5 vs 23.8 due to driver overhead), the assignment
    should still spread jobs evenly based on virtual tracking.
    """
    # Simulate GPUs with slightly different actual free memory
    gpu0 = GPUInfo(device_id=0, name="A30", total_memory_gb=24.0, free_memory_gb=23.5)
    gpu1 = GPUInfo(device_id=1, name="A30", total_memory_gb=24.0, free_memory_gb=23.8)

    gpus = [gpu0, gpu1]

    # All 3 jobs should NOT land on the same GPU
    result1 = assign_gpu(gpus, {})
    result2 = assign_gpu(gpus, {result1: 1})

    # Jobs must be on different GPUs
    assert result1 != result2


def test_query_gpus_no_nvidia_smi():
    """Test query_gpus returns empty list when nvidia-smi not found."""
    with patch("spatial_factorization.runner.subprocess.run", side_effect=FileNotFoundError):
        result = query_gpus()
        assert result == []


def test_query_gpus_parses_nvidia_smi():
    """Test query_gpus parses nvidia-smi CSV output."""
    fake_output = (
        "0, NVIDIA A30, 24576, 20480\n"
        "1, NVIDIA A30, 24576, 18432\n"
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_output

    with patch("spatial_factorization.runner.subprocess.run", return_value=mock_result):
        result = query_gpus()

        assert len(result) == 2
        assert result[0].device_id == 0
        assert result[0].name == "NVIDIA A30"
        assert result[0].total_memory_gb == pytest.approx(24.0, abs=0.1)
        assert result[0].free_memory_gb == pytest.approx(20.0, abs=0.1)
        assert result[1].device_id == 1
        assert result[1].free_memory_gb == pytest.approx(18.0, abs=0.1)


def test_job_creation():
    """Test Job dataclass creation."""
    job = Job(name="pnmf", config_path=Path("/tmp/pnmf.yaml"))
    assert job.name == "pnmf"
    assert job.status == "pending"
    assert job.device == "cpu"
    assert job.gpu_id is None
    assert job.pid is None
    assert job.elapsed == 0.0
    assert job.error is None


def test_run_status():
    """Test RunStatus dataclass."""
    status = RunStatus()
    assert status.start_time is not None
    assert status.end_time is None
    assert status.jobs == {}

    # Add a job
    status.jobs["pnmf"] = {"status": "completed", "elapsed": 100.0}

    # Test to_dict
    result = status.to_dict()
    assert "start_time" in result
    assert "jobs" in result
    assert result["jobs"]["pnmf"]["status"] == "completed"


def test_config_preprocessed_exists():
    """Test Config.preprocessed_exists() static method."""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # No preprocessed directory
        assert Config.preprocessed_exists(output_dir) is False

        # Create preprocessed directory without Y.npz
        (output_dir / "preprocessed").mkdir()
        assert Config.preprocessed_exists(output_dir) is False

        # Create Y.npz
        (output_dir / "preprocessed" / "Y.npz").touch()
        assert Config.preprocessed_exists(output_dir) is True


def test_job_runner_resolve_configs_per_model():
    """Test JobRunner._resolve_configs with a per-model config."""
    with TemporaryDirectory() as tmpdir:
        # Create a per-model config
        config_path = Path(tmpdir) / "pnmf.yaml"
        config = Config.from_dict({
            "name": "test_pnmf",
            "dataset": "test",
            "model": {
                "n_components": 10,
                "spatial": False,
                "prior": "GaussianPrior",
            },
        })
        config.save_yaml(config_path)

        runner = JobRunner(config_path)
        result = runner._resolve_configs()

        assert len(result) == 1
        assert result[0] == config_path


def test_job_runner_resolve_configs_general():
    """Test JobRunner._resolve_configs with a general config."""
    with TemporaryDirectory() as tmpdir:
        # Create a general config
        config_path = Path(tmpdir) / "general.yaml"
        config = Config.from_dict({
            "name": "test",
            "dataset": "test",
            "model": {
                "n_components": 10,
                # No spatial key = general config
            },
        })
        config.save_yaml(config_path)

        # Patch generate_configs to return known configs
        with patch("spatial_factorization.generate.generate_configs") as mock_gen:
            mock_gen.return_value = {
                "pnmf": Path(tmpdir) / "pnmf.yaml",
                "SVGP": Path(tmpdir) / "SVGP.yaml",
                "MGGP_SVGP": Path(tmpdir) / "MGGP_SVGP.yaml",
            }

            runner = JobRunner(config_path)
            result = runner._resolve_configs()

            assert len(result) == 3
            mock_gen.assert_called_once_with(config_path)


def test_job_runner_resolve_configs_directory():
    """Test JobRunner._resolve_configs with a directory."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create subdirectories with general.yaml files
        (tmpdir / "dataset1").mkdir()
        (tmpdir / "dataset2").mkdir()

        gen1 = tmpdir / "dataset1" / "general.yaml"
        gen2 = tmpdir / "dataset2" / "general.yaml"

        Config.from_dict({
            "name": "d1",
            "dataset": "dataset1",
            "model": {"n_components": 10},
        }).save_yaml(gen1)

        Config.from_dict({
            "name": "d2",
            "dataset": "dataset2",
            "model": {"n_components": 10},
        }).save_yaml(gen2)

        # Patch generate_configs to return per-model configs
        def mock_generate(path):
            dataset = path.parent.name
            return {
                "pnmf": path.parent / f"{dataset}_pnmf.yaml",
                "SVGP": path.parent / f"{dataset}_SVGP.yaml",
            }

        with patch("spatial_factorization.generate.generate_configs", side_effect=mock_generate):
            runner = JobRunner(tmpdir)
            result = runner._resolve_configs()

            # Should have 2 datasets x 2 models = 4 configs
            assert len(result) == 4


def test_job_runner_get_launch_env():
    """Test JobRunner._get_launch_env()."""
    runner = JobRunner(Path("/tmp/config.yaml"))

    # CPU
    env = runner._get_launch_env(None)
    assert "CUDA_VISIBLE_DEVICES" not in env

    # GPU
    env = runner._get_launch_env(0)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"

    env = runner._get_launch_env(1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
