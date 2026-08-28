"""CLI entrypoints."""

import importlib
import importlib.util

import click


@click.group()
def cli() -> None:
    """Execute CLI command."""


@click.command()
@click.argument("app_name")
def app(app_name: str) -> None:
    """Start app by name."""
    importlib.import_module(f"clinlp_apps.{app_name}.app").app.run()


@cli.group()
def install() -> None:
    """Install optional runtime assets."""


@install.command("transformers")
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download of model files even if they are cached.",
)
def install_transformers(force_download: bool) -> None:
    """Pre-download transformers used by clinlp components."""
    if importlib.util.find_spec("transformers") is None:
        msg = (
            'The "transformers" extra is not installed. '
            'Install it first using: pip install "clinlp[transformers]"'
        )
        raise click.ClickException(msg)

    transformer = importlib.import_module("clinlp.ie.qualifier.transformer")

    for model_cls in (
        transformer.NegationTransformer,
        transformer.ExperiencerTransformer,
        transformer.PiiTransformer,
        transformer.CardioTransformer,
    ):
        click.echo(
            f"Prefetching {model_cls.PRETRAINED_MODEL_NAME_OR_PATH}@{model_cls.REVISION}"
        )
        model_cls.prefetch(force_download=force_download)

    click.echo("Done prefetching transformer models.")


cli.add_command(app)

if __name__ == "__main__":
    cli()
