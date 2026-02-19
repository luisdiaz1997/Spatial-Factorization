"""CLI entry point for spatial_factorization using Click."""
import click

STAGE_ORDER = ["preprocess", "train", "analyze", "figures"]


def _run_stage(stage, config):
    """Import and run a single pipeline stage."""
    if stage == "preprocess":
        from .commands import preprocess as cmd
    elif stage == "train":
        from .commands import train as cmd
    elif stage == "analyze":
        from .commands import analyze as cmd
    elif stage == "figures":
        from .commands import figures as cmd
    else:
        raise click.BadParameter(f"Unknown stage: {stage}")
    cmd.run(config)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Spatial transcriptomics factorization toolkit."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def preprocess(config):
    """Preprocess dataset into standardized format (run once)."""
    _run_stage("preprocess", config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--resume", is_flag=True, default=False, help="Resume training from saved checkpoint, appending to ELBO history")
def train(config, resume):
    """Train a PNMF model."""
    from .commands import train as train_cmd
    train_cmd.run(config, resume=resume)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def analyze(config):
    """Analyze a trained model (Moran's I, reconstruction, etc.)."""
    _run_stage("analyze", config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def figures(config):
    """Generate publication figures."""
    _run_stage("figures", config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to general config YAML")
def generate(config):
    """Generate per-model configs from a general.yaml.

    \b
    Example:
        spatial_factorization generate -c configs/slideseq/general.yaml
    """
    from .generate import generate_configs

    generated = generate_configs(config)
    click.echo(f"Generated {len(generated)} model configs:")
    for name, path in generated.items():
        click.echo(f"  {name}: {path}")


@cli.command("run")
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML or directory")
@click.option("--force", is_flag=True, help="Force re-run preprocessing")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.option("--resume", is_flag=True, help="Resume training: resumes models with a checkpoint, trains new ones from scratch")
@click.option("--config-name", default="general.yaml", show_default=True, help="Config filename to search for when config is a directory (e.g. general_test.yaml)")
def run_pipeline(stages, config, force, dry_run, resume, config_name):
    """Run pipeline stages sequentially, or run all models in parallel.

    \b
    STAGES:
        preprocess, train, analyze, figures  - Run specific stages sequentially
        all                                   - Run full pipeline (preprocess → train → analyze → figures)

    \b
    CONFIG BEHAVIOR (with 'all' stage):
        - Per-model YAML (e.g., svgp.yaml):     Runs all stages for that single model
        - General YAML (e.g., general.yaml):     Generates per-model configs, then runs all in parallel
        - Directory:                             Recursively finds all general.yaml files, generates configs, runs all in parallel

    \b
    EXAMPLES:
        # Single model, all stages
        spatial_factorization run all -c configs/slideseq/MGGP_SVGP.yaml

        # Single model, specific stages
        spatial_factorization run train analyze -c configs/slideseq/MGGP_SVGP.yaml

        # All models from general config (parallel)
        spatial_factorization run all -c configs/slideseq/general.yaml

        # All models from all datasets (parallel)
        spatial_factorization run all -c configs/

        # All datasets using test configs
        spatial_factorization run all -c configs/ --config-name general_test.yaml

        # Resume training across all models (trains new ones from scratch)
        spatial_factorization run all -c configs/slideseq/general.yaml --resume

        # Dry run to see plan
        spatial_factorization run all -c configs/slideseq/general.yaml --dry-run

        # Force re-preprocessing
        spatial_factorization run all -c configs/slideseq/general.yaml --force
    """
    stages = [s.lower() for s in stages]

    # Check for 'all' stage - invokes multiplex runner
    if "all" in stages:
        from .runner import JobRunner

        JobRunner(config, force_preprocess=force, dry_run=dry_run, resume=resume, config_name=config_name).run()
        return

    # Validate stages
    valid_stages = set(STAGE_ORDER)
    invalid = set(stages) - valid_stages
    if invalid:
        raise click.BadParameter(f"Unknown stage(s): {', '.join(invalid)}. Valid: {', '.join(STAGE_ORDER)}, or 'all'")

    # Sort stages into pipeline order
    ordered = [s for s in STAGE_ORDER if s in stages]
    click.echo(f"Running stages: {' → '.join(ordered)}")
    for stage in ordered:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Stage: {stage}")
        click.echo(f"{'='*60}\n")
        if stage == "train" and resume:
            from .commands import train as train_cmd
            train_cmd.run(config, resume=True)
        else:
            _run_stage(stage, config)
    click.echo(f"\nAll stages complete.")


if __name__ == "__main__":
    cli()
