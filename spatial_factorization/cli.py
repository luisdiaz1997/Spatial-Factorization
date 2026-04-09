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
@click.option("--video", is_flag=True, default=False, help="Capture factor snapshots during training and save as MP4.")
def train(config, resume, video):
    """Train a PNMF model."""
    from .commands import train as train_cmd
    train_cmd.run(config, resume=resume, video=video)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--probabilistic", is_flag=True, default=False,
              help="Override saved KNN strategy and use probabilistic neighbor sampling for this analyze run")
def analyze(config, probabilistic):
    """Analyze a trained model (Moran's I, reconstruction, etc.)."""
    from .commands import analyze as cmd
    cmd.run(config, probabilistic=probabilistic)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--no-heatmap", is_flag=True, default=False, help="Skip celltype_gene_loadings and factor_gene_loadings heatmaps")
def figures(config, no_heatmap):
    """Generate publication figures."""
    from .commands import figures as cmd
    cmd.run(config, no_heatmap=no_heatmap)


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


@cli.command("multianalyze")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to any config YAML for the dataset (used to resolve output_dir)")
@click.argument("models", nargs=-1, required=True)
@click.option("--n-pairs", default=2, show_default=True,
              help="Number of matched pairs to show (only used when exactly 2 models given)")
@click.option("--match-against", default=None,
              help="Model to use for the initial reference matching (3+ model case; default: second model)")
@click.option("--output", "-o", default=None,
              help="Output directory (default: output_dir/figures/)")
def multianalyze(config, models, n_pairs, match_against, output):
    """Compare matched factors across trained models.

    \b
    With 2 models: finds top --n-pairs matched factor pairs and shows each pair
    as [2D | 3D] side by side, model A on row 0 and model B on row 1.

    With 3+ models: finds the single best match between model 0 (reference) and
    model 1, then matches that factor against all remaining models. Plots all
    models side by side with 2D on the top row and 3D on the bottom row.

    \b
    EXAMPLES:
        spatial_factorization multianalyze -c configs/slideseq/general.yaml svgp mggp_svgp
        spatial_factorization multianalyze -c configs/slideseq/general.yaml svgp mggp_svgp --n-pairs 3
        spatial_factorization multianalyze -c configs/slideseq/general.yaml \\
            svgp mggp_svgp pnmf lcgp mggp_lcgp
    """
    from .commands.multianalyze import run as _run
    _run(config, list(models), n_pairs=n_pairs, match_against=match_against, output_path=output)


@cli.command("run")
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML or directory")
@click.option("--force", is_flag=True, help="Force re-run preprocessing")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.option("--resume", is_flag=True, help="Resume training: resumes models with a checkpoint, trains new ones from scratch")
@click.option("--video", is_flag=True, default=False, help="Capture factor snapshots during training and save as GIF.")
@click.option("--gpu-only", is_flag=True, default=False, help="Only assign jobs to GPUs; never fall back to CPU.")
@click.option("--config-name", default="general.yaml", show_default=True, help="Config filename to search for when config is a directory (e.g. general_test.yaml)")
@click.option("--failed", "failed_only", is_flag=True, help="Re-run only jobs that failed in the previous run (reads run_status.json)")
@click.option("--no-heatmap", is_flag=True, default=False, help="Skip celltype_gene_loadings and factor_gene_loadings heatmaps")
@click.option("--skip-general", is_flag=True, default=False, help="Skip general configs; run all non-general yamls in the directory as per-model configs")
@click.option("--probabilistic", is_flag=True, default=False, help="Override saved KNN strategy in analyze stage to use probabilistic neighbor sampling")
def run_pipeline(stages, config, force, dry_run, resume, video, gpu_only, config_name, failed_only, no_heatmap, skip_general, probabilistic):
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

        # Re-run only failed jobs from the previous run
        spatial_factorization run all -c configs/ --failed

        # Dry run to see plan
        spatial_factorization run all -c configs/slideseq/general.yaml --dry-run

        # Force re-preprocessing
        spatial_factorization run all -c configs/slideseq/general.yaml --force
    """
    stages = [s.lower() for s in stages]

    # Check for 'all' stage - invokes multiplex runner
    if "all" in stages:
        from .runner import JobRunner

        JobRunner(config, force_preprocess=force, dry_run=dry_run, resume=resume, config_name=config_name, failed_only=failed_only, video=video, gpu_only=gpu_only, no_heatmap=no_heatmap, skip_general=skip_general, probabilistic=probabilistic).run()
        return

    # Validate stages
    valid_stages = set(STAGE_ORDER)
    invalid = set(stages) - valid_stages
    if invalid:
        raise click.BadParameter(f"Unknown stage(s): {', '.join(invalid)}. Valid: {', '.join(STAGE_ORDER)}, or 'all'")

    # Sort stages into pipeline order
    ordered = [s for s in STAGE_ORDER if s in stages]

    # If config is a directory or general.yaml, fan out through the multiplexer
    from pathlib import Path as _Path
    from .config import Config as _Config
    config_path = _Path(config)
    if config_path.is_dir() or _Config.is_general_config(config_path):
        from .runner import JobRunner
        JobRunner(config, stages=ordered, force_preprocess=force, dry_run=dry_run, resume=resume, config_name=config_name, failed_only=failed_only, video=video, gpu_only=gpu_only, no_heatmap=no_heatmap, skip_general=skip_general, probabilistic=probabilistic).run()
        return

    # Existing sequential path (single per-model config, no multiplexer)
    click.echo(f"Running stages: {' → '.join(ordered)}")
    for stage in ordered:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Stage: {stage}")
        click.echo(f"{'='*60}\n")
        if stage == "train":
            from .commands import train as train_cmd
            train_cmd.run(config, resume=resume, video=video)
        elif stage == "figures":
            from .commands import figures as figures_cmd
            figures_cmd.run(config, no_heatmap=no_heatmap)
        elif stage == "analyze":
            from .commands import analyze as analyze_cmd
            analyze_cmd.run(config, probabilistic=probabilistic)
        else:
            _run_stage(stage, config)
    click.echo(f"\nAll stages complete.")


if __name__ == "__main__":
    cli()
