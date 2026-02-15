"""Configuration system for spatial factorization experiments.

Simple two-level config: top-level fields + nested dicts for model/training/output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """Experiment configuration.

    Attributes:
        name: Experiment name
        seed: Random seed
        dataset: Dataset name (e.g., 'slideseq')
        preprocessing: Dataset preprocessing parameters
        model: Model parameters dict
        training: Training parameters dict
        output_dir: Output directory path
    """

    name: str
    seed: int = 42
    dataset: str = "slideseq"

    # Preprocessing config (dataset-specific)
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "spatial_scale": 50.0,
        "filter_mt": True,
        "min_counts": 100,
        "min_cells": 10,
    })

    # Model config (passed to PNMF)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "n_components": 10,
        "spatial": False,
        "prior": "GaussianPrior",  # GaussianPrior, SVGP, VNNGP, LCGP
        "loadings_mode": "projected",
        "mode": "expanded",
        "training_mode": "standard",
        "E": 3,
    })

    # Training config (passed to PNMF)
    training: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 10000,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "tol": 1e-4,
        "verbose": False,
        "device": "cpu",
        "batch_size": None,
        "y_batch_size": None,
        "shuffle": True,
    })

    # Output config
    output_dir: str = "outputs"

    @property
    def prior(self) -> str:
        """Return the prior class name."""
        return self.model.get("prior", "GaussianPrior")

    @property
    def spatial(self) -> bool:
        """Return whether spatial mode is enabled."""
        return self.model.get("spatial", False)

    @property
    def groups(self) -> bool:
        """Return whether multi-group (MGGP) mode is enabled."""
        return self.model.get("groups", False)

    @property
    def local(self) -> bool:
        """Return whether LCGP (local conditioning) mode is enabled."""
        return self.model.get("local", False)

    @property
    def model_name(self) -> str:
        """Return model directory name based on spatial/groups config.

        - Non-spatial: "pnmf"
        - Spatial, no groups: "{prior}" e.g. "svgp"
        - Spatial, with groups: "mggp_{prior}" e.g. "mggp_svgp"
        """
        if not self.spatial:
            return "pnmf"
        prior = self.model.get("prior", "SVGP").lower()
        if self.groups:
            return f"mggp_{prior}"
        return prior

    def to_pnmf_kwargs(self) -> Dict[str, Any]:
        """Merge model and training configs for PNMF constructor."""
        kwargs = {}

        # Model params
        kwargs["n_components"] = self.model.get("n_components", 10)
        kwargs["loadings_mode"] = self.model.get("loadings_mode", "projected")
        kwargs["mode"] = self.model.get("mode", "expanded")
        kwargs["training_mode"] = self.model.get("training_mode", "standard")
        kwargs["E"] = self.model.get("E", 3)

        # Spatial params (SVGP, VNNGP, LCGP)
        if self.model.get("spatial", False):
            kwargs["spatial"] = True
            # Note: prior is NOT passed to PNMF - it auto-selects based on spatial/multigroup/local
            kwargs["kernel"] = self.model.get("kernel", "Matern32")
            kwargs["multigroup"] = self.model.get("groups", False)
            kwargs["lengthscale"] = float(self.model.get("lengthscale", 1.0))
            kwargs["sigma"] = float(self.model.get("sigma", 1.0))
            kwargs["group_diff_param"] = float(self.model.get("group_diff_param", 10.0))
            kwargs["train_lengthscale"] = self.model.get("train_lengthscale", False)

            # LCGP vs SVGP params
            is_local = self.model.get("local", False)
            kwargs["local"] = is_local

            if is_local:
                # LCGP-specific params
                kwargs["K"] = self.model.get("K", 50)
                kwargs["rank"] = self.model.get("rank", None)
                kwargs["low_rank_mode"] = self.model.get("low_rank_mode", "softplus")
                kwargs["precompute_knn"] = self.model.get("precompute_knn", True)
            else:
                # SVGP-specific params (not used by LCGP)
                kwargs["num_inducing"] = self.model.get("num_inducing", 3000)
                kwargs["cholesky_mode"] = self.model.get("cholesky_mode", "exp")
                kwargs["diagonal_only"] = self.model.get("diagonal_only", False)
                kwargs["inducing_allocation"] = self.model.get("inducing_allocation", "proportional")

        # Training params
        kwargs["max_iter"] = self.training.get("max_iter", 10000)
        kwargs["learning_rate"] = float(self.training.get("learning_rate", 0.01))
        kwargs["optimizer"] = self.training.get("optimizer", "Adam")
        kwargs["tol"] = float(self.training.get("tol", 1e-4))
        kwargs["verbose"] = self.training.get("verbose", False)

        # Device: map 'gpu' to 'auto'
        device = self.training.get("device", "cpu")
        kwargs["device"] = "auto" if device == "gpu" else device

        # Batching
        kwargs["batch_size"] = self.training.get("batch_size") or self.training.get("x_batch")
        kwargs["y_batch_size"] = self.training.get("y_batch_size") or self.training.get("y_batch")
        kwargs["shuffle"] = self.training.get("shuffle", True)

        return kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            name=data["name"],
            seed=data.get("seed", 42),
            dataset=data.get("dataset", "slideseq"),
            preprocessing=data.get("preprocessing", {}),
            model=data.get("model", {}),
            training=data.get("training", {}),
            output_dir=data.get("output_dir", data.get("output", {}).get("dir", "outputs")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "model": self.model,
            "training": self.training,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file with blank lines between sections."""
        d = self.to_dict()
        with open(path, "w") as f:
            # Write simple fields in order
            f.write(f"name: {d['name']}\n")
            f.write(f"seed: {d['seed']}\n")
            f.write(f"dataset: {d['dataset']}\n")
            f.write(f"output_dir: {d['output_dir']}\n")

            # Write sections with blank lines before each
            for key in ["preprocessing", "model", "training"]:
                f.write(f"\n{key}:\n")
                # Get yaml content without trailing newline from dump
                content = yaml.dump(d[key], default_flow_style=False, sort_keys=False)
                # Indent each line by 2 spaces
                for line in content.strip().split('\n'):
                    f.write(f"  {line}\n")

    @classmethod
    def is_general_config(cls, path: str | Path) -> bool:
        """Check if a config is a general config (no model.spatial key).

        A general config is a superset of all model params and will be used
        to generate per-model configs (pnmf.yaml, svgp.yaml, mggp_svgp.yaml).

        Args:
            path: Path to the YAML config file.

        Returns:
            True if the config is general (no model.spatial key), False otherwise.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        model_section = data.get("model", {})
        return "spatial" not in model_section

    @staticmethod
    def preprocessed_exists(output_dir: str | Path) -> bool:
        """Check if preprocessed data exists in the output directory.

        Args:
            output_dir: Path to the output directory.

        Returns:
            True if {output_dir}/preprocessed/Y.npz exists, False otherwise.
        """
        preprocessed_path = Path(output_dir) / "preprocessed" / "Y.npz"
        return preprocessed_path.exists()
