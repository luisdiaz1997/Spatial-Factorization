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
    def model_name(self) -> str:
        """Return model directory name based on spatial/groups config.

        - Non-spatial: "pnmf"
        - Spatial, no groups: "{prior}" e.g. "SVGP"
        - Spatial, with groups: "MGGP_{prior}" e.g. "MGGP_SVGP"
        """
        if not self.spatial:
            return "pnmf"
        prior = self.model.get("prior", "SVGP")
        if self.groups:
            return f"MGGP_{prior}"
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
            kwargs["prior"] = self.model.get("prior", "SVGP")
            kwargs["kernel"] = self.model.get("kernel", "Matern32")
            kwargs["multigroup"] = self.model.get("groups", False)
            kwargs["num_inducing"] = self.model.get("num_inducing", 3000)
            kwargs["lengthscale"] = float(self.model.get("lengthscale", 1.0))
            kwargs["sigma"] = float(self.model.get("sigma", 1.0))
            kwargs["group_diff_param"] = float(self.model.get("group_diff_param", 10.0))
            kwargs["train_lengthscale"] = self.model.get("train_lengthscale", False)
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
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
