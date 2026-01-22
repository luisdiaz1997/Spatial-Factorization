"""CLI entry point for spatial_factorization using Click."""
import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Spatial transcriptomics factorization toolkit."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def preprocess(config):
    """Preprocess dataset into standardized format (run once)."""
    from .commands import preprocess as preprocess_cmd

    preprocess_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def train(config):
    """Train a PNMF model."""
    from .commands import train as train_cmd

    train_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def analyze(config):
    """Analyze a trained model (Moran's I, reconstruction, etc.)."""
    from .commands import analyze as analyze_cmd

    analyze_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def figures(config):
    """Generate publication figures."""
    from .commands import figures as figures_cmd

    figures_cmd.run(config)


if __name__ == "__main__":
    cli()
