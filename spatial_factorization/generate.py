"""Config generation from general.yaml to per-model YAMLs.

Reads a general config (superset of all model params) and generates
per-model configs (pnmf.yaml, SVGP.yaml, MGGP_SVGP.yaml) with proper
field filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import Config


# Model variants to generate from a general config
MODEL_VARIANTS = [
    {"name": "pnmf", "spatial": False, "groups": False, "prior": "GaussianPrior"},
    {"name": "svgp", "spatial": True, "groups": False, "prior": "SVGP"},
    {"name": "mggp_svgp", "spatial": True, "groups": True, "prior": "SVGP"},
]

# Field categories for filtering
COMMON_MODEL_FIELDS = {
    "n_components",
    "mode",
    "loadings_mode",
    "training_mode",
    "E",
}

SPATIAL_MODEL_FIELDS = {
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

GROUPS_MODEL_FIELDS = {
    "groups",
    "group_diff_param",
    "inducing_allocation",
}


def generate_configs(general_path: str | Path) -> Dict[str, Path]:
    """Generate per-model configs from a general.yaml.

    Args:
        general_path: Path to the general config YAML.

    Returns:
        Dictionary mapping model name (e.g., 'pnmf', 'SVGP', 'MGGP_SVGP')
        to the generated config file path.

    Raises:
        ValueError: If general_path is not a general config.
    """
    general_path = Path(general_path)

    # Verify it's a general config
    if not Config.is_general_config(general_path):
        raise ValueError(f"{general_path} is not a general config (must lack model.spatial)")

    # Load general config
    config = Config.from_yaml(general_path)

    # Determine output directory for generated configs
    # Use same directory as general.yaml, or fallback to configs/{dataset}/
    if general_path.parent.name == config.dataset:
        output_dir = general_path.parent
    else:
        output_dir = Path("configs") / config.dataset
        output_dir.mkdir(parents=True, exist_ok=True)

    generated = {}

    for variant in MODEL_VARIANTS:
        model_config = _generate_model_config(config, variant)

        # Set name to {dataset}_{model_name}
        model_config.name = f"{config.dataset}_{variant['name'].lower()}"

        # Determine output filename
        output_path = output_dir / f"{variant['name']}.yaml"

        # Save config
        model_config.save_yaml(output_path)
        generated[variant["name"]] = output_path

    return generated


def _generate_model_config(
    config: Config, variant: Dict[str, any]
) -> Config:
    """Generate a Config for a specific model variant.

    Filters the model section to only include relevant fields for the variant.

    Args:
        config: The general config.
        variant: Dict with 'name', 'spatial', 'groups', 'prior' keys.

    Returns:
        A new Config with filtered model section.
    """
    is_spatial = variant["spatial"]
    is_groups = variant["groups"]

    # Start with common model fields
    model_dict = {}

    for key in COMMON_MODEL_FIELDS:
        if key in config.model:
            model_dict[key] = config.model[key]

    # Add spatial fields if applicable
    if is_spatial:
        model_dict["spatial"] = True
        model_dict["prior"] = variant["prior"]
        for key in SPATIAL_MODEL_FIELDS:
            if key in config.model and key != "spatial":
                model_dict[key] = config.model[key]

        # Add group fields if applicable
        if is_groups:
            model_dict["groups"] = True
            for key in GROUPS_MODEL_FIELDS:
                if key in config.model and key != "groups":
                    model_dict[key] = config.model[key]
    else:
        model_dict["spatial"] = False
        model_dict["prior"] = "GaussianPrior"

    # Build new config dict
    config_dict = {
        "name": config.name,
        "seed": config.seed,
        "dataset": config.dataset,
        "preprocessing": config.preprocessing,
        "model": model_dict,
        "training": config.training,
        "output_dir": config.output_dir,
    }

    return Config.from_dict(config_dict)
