"""Configuration system for spatial factorization experiments.

Configs store experiment settings for reproducibility. Model construction
and training is handled by the PNMF package.

Example usage:
    from spatial_factorization import Config, load_dataset
    from PNMF import PNMF

    # Load config and data
    config = Config.from_yaml("configs/slideseq/lcgp.yaml")
    data = load_dataset(config.dataset)

    # Build model using PNMF (when spatial=True is implemented)
    model = PNMF(
        n_components=config.model.n_components,
        spatial=True,
        gp_class=config.model.gp_class,
        **config.model.to_pnmf_kwargs()
    )
    model.fit(data.Y.T.numpy(), X=data.X.numpy())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str
    spatial_scale: float = 50.0
    filter_mt: bool = True
    min_counts: int = 100
    min_cells: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(**data)


@dataclass
class ModelConfig:
    """Model configuration (stored for reproducibility, used with PNMF API)."""

    n_components: int = 10
    spatial: bool = True
    gp_class: str = "LCGP"  # SVGP, VNNGP, LCGP

    # GP hyperparameters
    lengthscale: float = 4.0
    sigma: float = 1.0
    jitter: float = 1e-5

    # VNNGP/LCGP specific
    K: Optional[int] = 50

    # SVGP specific
    n_inducing: Optional[int] = None

    # Loadings
    loadings_mode: str = "projected"

    # ELBO mode
    mode: str = "expanded"  # simple, expanded, lower-bound

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)

    def to_pnmf_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for PNMF constructor."""
        kwargs = {
            "n_components": self.n_components,
            "spatial": self.spatial,
            "mode": self.mode,
            "loadings_mode": self.loadings_mode,
        }
        if self.spatial:
            kwargs["gp_class"] = self.gp_class
            kwargs["lengthscale"] = self.lengthscale
            if self.K is not None:
                kwargs["K"] = self.K
            if self.n_inducing is not None:
                kwargs["n_inducing"] = self.n_inducing
        return kwargs


@dataclass
class TrainingConfig:
    """Training configuration."""

    max_iter: int = 10000
    learning_rate: float = 0.01
    optimizer: str = "Adam"
    tol: float = 1e-4
    verbose: bool = True
    device: str = "cpu"  # 'cpu', 'gpu' (auto-selects cuda/mps), 'cuda', 'mps'

    # Batch training (mini-batch for large datasets)
    x_batch: Optional[int] = None  # Mini-batch size for samples (N dimension)
    y_batch: Optional[int] = None  # Mini-batch size for features (D dimension)
    shuffle: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        # Accept both x_batch/batch_size and y_batch/y_batch_size for compatibility
        # but normalize to x_batch/y_batch internally
        return cls(
            max_iter=int(data.get("max_iter", 10000)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            optimizer=data.get("optimizer", "Adam"),
            tol=float(data.get("tol", 1e-4)),
            verbose=data.get("verbose", True),
            device=data.get("device", "cpu"),
            x_batch=data.get("x_batch") or data.get("batch_size"),
            y_batch=data.get("y_batch") or data.get("y_batch_size"),
            shuffle=data.get("shuffle", True),
        )

    def to_pnmf_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for PNMF constructor.

        Maps our naming convention (x_batch/y_batch) to PNMF's (batch_size/y_batch_size).
        """
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "tol": self.tol,
            "verbose": self.verbose,
            "device": "auto" if self.device == "gpu" else self.device,
            "batch_size": self.x_batch,
            "y_batch_size": self.y_batch,
            "shuffle": self.shuffle,
        }


@dataclass
class OutputConfig:
    """Output configuration."""

    dir: str = "outputs"
    save_checkpoint: bool = True
    save_history: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(**data)


@dataclass
class Config:
    """Full experiment configuration."""

    name: str
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="slideseq"))
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            name=data["name"],
            seed=data.get("seed", 42),
            dataset=DatasetConfig.from_dict(data.get("dataset", {"name": "slideseq"})),
            model=ModelConfig.from_dict(data.get("model", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @property
    def output_dir(self) -> Path:
        """Get output directory as Path."""
        return Path(self.output.dir)
