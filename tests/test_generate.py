"""Tests for config generation from general.yaml."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from spatial_factorization.config import Config
from spatial_factorization.generate import (
    COMMON_MODEL_FIELDS,
    generate_configs,
    GROUPS_MODEL_FIELDS,
    SPATIAL_MODEL_FIELDS,
    _generate_model_config,
)


def test_common_model_fields_set():
    """Verify that common model fields are defined."""
    assert COMMON_MODEL_FIELDS == {
        "n_components",
        "mode",
        "loadings_mode",
        "training_mode",
        "E",
    }


def test_spatial_model_fields_set():
    """Verify that spatial model fields are defined."""
    expected = {
        "spatial",
        "prior",
        "kernel",
        "num_inducing",
        "lengthscale",
        "sigma",
        "train_lengthscale",
        "cholesky_mode",
        "diagonal_only",
    }
    assert SPATIAL_MODEL_FIELDS == expected


def test_groups_model_fields_set():
    """Verify that groups model fields are defined."""
    assert GROUPS_MODEL_FIELDS == {
        "groups",
        "group_diff_param",
        "inducing_allocation",
    }


@pytest.fixture
def general_config_dict():
    """Return a minimal general config dict."""
    return {
        "name": "test",
        "seed": 42,
        "dataset": "test",
        "preprocessing": {"spatial_scale": 50.0},
        "model": {
            "n_components": 10,
            "mode": "expanded",
            "loadings_mode": "multiplicative",
            "training_mode": "standard",
            "E": 3,
            "kernel": "Matern32",
            "num_inducing": 3000,
            "lengthscale": 8.0,
            "sigma": 1.0,
            "train_lengthscale": False,
            "cholesky_mode": "exp",
            "diagonal_only": False,
            "group_diff_param": 1.0,
            "inducing_allocation": "derived",
        },
        "training": {
            "max_iter": 1000,
            "learning_rate": 0.01,
            "device": "cpu",
        },
        "output_dir": "outputs/test",
    }


def test_generate_pnmf_config(general_config_dict):
    """Test generating a pnmf config from a general config."""
    general_config = Config.from_dict(general_config_dict)
    variant = {"name": "pnmf", "spatial": False, "groups": False, "prior": "GaussianPrior"}

    result = _generate_model_config(general_config, variant)

    # Check basic properties
    assert result.spatial is False
    assert result.prior == "GaussianPrior"

    # Check only common fields present
    model_keys = set(result.model.keys())
    assert model_keys == {"spatial", "prior"} | COMMON_MODEL_FIELDS

    # Check values
    assert result.model["n_components"] == 10
    assert result.model["spatial"] is False


def test_generate_svgp_config(general_config_dict):
    """Test generating an SVGP config from a general config."""
    general_config = Config.from_dict(general_config_dict)
    variant = {"name": "svgp", "spatial": True, "groups": False, "prior": "SVGP"}

    result = _generate_model_config(general_config, variant)

    # Check basic properties
    assert result.spatial is True
    assert result.groups is False
    assert result.prior == "SVGP"

    # Check spatial fields present, but no group fields
    model_keys = set(result.model.keys())
    assert "groups" not in model_keys
    assert "group_diff_param" not in model_keys
    assert "inducing_allocation" not in model_keys

    # Check spatial values
    assert result.model["kernel"] == "Matern32"
    assert result.model["num_inducing"] == 3000


def test_generate_mggp_svgp_config(general_config_dict):
    """Test generating an mggp_svgp config from a general config."""
    general_config = Config.from_dict(general_config_dict)
    variant = {"name": "mggp_svgp", "spatial": True, "groups": True, "prior": "SVGP"}

    result = _generate_model_config(general_config, variant)

    # Check basic properties
    assert result.spatial is True
    assert result.groups is True
    assert result.prior == "SVGP"

    # Check all fields present
    model_keys = set(result.model.keys())
    assert "groups" in model_keys
    assert "group_diff_param" in model_keys
    assert "inducing_allocation" in model_keys

    # Check values
    assert result.model["groups"] is True
    assert result.model["group_diff_param"] == 1.0
    assert result.model["kernel"] == "Matern32"


def test_generate_configs_from_file():
    """Test generating configs from a general.yaml file."""
    with TemporaryDirectory() as tmpdir:
        # Create general.yaml
        general_path = Path(tmpdir) / "general.yaml"
        general_config = Config.from_dict({
            "name": "test",
            "seed": 42,
            "dataset": "test",
            "preprocessing": {"spatial_scale": 50.0},
            "model": {
                "n_components": 10,
                "mode": "expanded",
                "loadings_mode": "multiplicative",
                "training_mode": "standard",
                "E": 3,
                "kernel": "Matern32",
                "num_inducing": 3000,
                "lengthscale": 8.0,
                "sigma": 1.0,
                "train_lengthscale": False,
                "cholesky_mode": "exp",
                "diagonal_only": False,
                "group_diff_param": 1.0,
                "inducing_allocation": "derived",
            },
            "training": {
                "max_iter": 1000,
                "learning_rate": 0.01,
                "device": "cpu",
            },
            "output_dir": "outputs/test",
        })
        general_config.save_yaml(general_path)

        # Generate configs
        generated = generate_configs(general_path)

        # Check results - now includes LCGP variants
        assert set(generated.keys()) == {"pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"}

        # Load each generated config and verify
        for name, path in generated.items():
            config = Config.from_yaml(path)
            assert config.name == f"test_{name}"

            if name == "pnmf":
                assert config.spatial is False
                assert config.prior == "GaussianPrior"
            elif name == "svgp":
                assert config.spatial is True
                assert config.groups is False
                assert config.local is False
                assert config.prior == "SVGP"
            elif name == "mggp_svgp":
                assert config.spatial is True
                assert config.groups is True
                assert config.local is False
                assert config.prior == "SVGP"
            elif name == "lcgp":
                assert config.spatial is True
                assert config.groups is False
                assert config.local is True
                assert config.prior == "LCGP"
            elif name == "mggp_lcgp":
                assert config.spatial is True
                assert config.groups is True
                assert config.local is True
                assert config.prior == "LCGP"


def test_generate_configs_rejects_non_general_config():
    """Test that generate_configs raises ValueError for non-general config."""
    with TemporaryDirectory() as tmpdir:
        # Create a non-general config (has spatial key)
        config_path = Path(tmpdir) / "pnmf.yaml"
        config = Config.from_dict({
            "name": "test_pnmf",
            "seed": 42,
            "dataset": "test",
            "preprocessing": {"spatial_scale": 50.0},
            "model": {
                "n_components": 10,
                "spatial": False,  # This makes it non-general
                "prior": "GaussianPrior",
            },
            "training": {"max_iter": 1000},
            "output_dir": "outputs/test",
        })
        config.save_yaml(config_path)

        # Should raise ValueError
        with pytest.raises(ValueError, match="not a general config"):
            generate_configs(config_path)


def test_config_is_general_config():
    """Test Config.is_general_config() method."""
    with TemporaryDirectory() as tmpdir:
        # General config (no spatial key)
        general_path = Path(tmpdir) / "general.yaml"
        Config.from_dict({
            "name": "test",
            "dataset": "test",
            "model": {
                "n_components": 10,
                # No "spatial" key
            },
        }).save_yaml(general_path)
        assert Config.is_general_config(general_path) is True

        # Non-general config (has spatial key)
        pnmf_path = Path(tmpdir) / "pnmf.yaml"
        Config.from_dict({
            "name": "test_pnmf",
            "dataset": "test",
            "model": {
                "n_components": 10,
                "spatial": False,  # Has spatial key
            },
        }).save_yaml(pnmf_path)
        assert Config.is_general_config(pnmf_path) is False
