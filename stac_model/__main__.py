import json

import typer
from rich.console import Console

from stac_model import __version__
from stac_model.examples import eurosat_resnet
from stac_model.schema import ItemMLModelExtension

app = typer.Typer(
    name="stac-model",
    help="A PydanticV2 validation and serialization library for the STAC Machine Learning Model Extension",
    add_completion=False,
)
console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]stac-model[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()


@app.command(name="")
def main(
    print_version: bool = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Prints the version of the stac-model package.",
    ),
) -> ItemMLModelExtension:
    """Generate example spec."""
    ml_model_meta = eurosat_resnet()
    with open("example.json", "w") as json_file:
        json.dump(ml_model_meta.item.to_dict(), json_file, indent=4)
    print("Example model metadata written to ./example.json.")
    return ml_model_meta


if __name__ == "__main__":
    app()
