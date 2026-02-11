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
def train(config):
    """Train a PNMF model."""
    _run_stage("train", config)


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


@cli.command("run")
@click.argument("stages", nargs=-1, required=True,
                type=click.Choice(STAGE_ORDER, case_sensitive=False))
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def run_pipeline(stages, config):
    """Run multiple stages sequentially.

    \b
    Examples:
        spatial_factorization run train analyze figures -c config.yaml
        spatial_factorization run preprocess train analyze figures -c config.yaml
    """
    # Sort stages into pipeline order
    ordered = [s for s in STAGE_ORDER if s in stages]
    click.echo(f"Running stages: {' → '.join(ordered)}")
    for stage in ordered:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Stage: {stage}")
        click.echo(f"{'='*60}\n")
        _run_stage(stage, config)
    click.echo(f"\nAll stages complete.")


if __name__ == "__main__":
    cli()
