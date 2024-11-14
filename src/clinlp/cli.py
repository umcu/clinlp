"""CLI entrypoints."""

import importlib

import click


@click.group()
def cli() -> None:
    """Execute CLI command."""


@click.command()
@click.argument("app_name")
def app(app_name: str) -> None:
    """Start app by name."""
    importlib.import_module(f"clinlp_apps.{app_name}.app").app.run_server()


cli.add_command(app)

if __name__ == "__main__":
    cli()
