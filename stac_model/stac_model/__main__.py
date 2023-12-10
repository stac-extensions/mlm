import typer
from rich.console import Console

from stac_model import __version__
from stac_model.schema import *


app = typer.Typer(
    name="stac-model",
    help="A PydanticV2 validation and serialization libary for the STAC ML Model Extension",
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
) -> None:
    """Generate example spec."""

    input_sig = TensorSignature(
        name="input_tensor", dtype="float32", shape=(-1, 13, 64, 64)
    )
    output_sig = TensorSignature(name="output_tensor", dtype="float32", shape=(-1, 10))
    model_sig = ModelSignature(inputs=[input_sig], outputs=[output_sig])
    model_artifact = ModelArtifact(path="s3://example/s3/uri/model.pt")
    class_map = ClassMap(
        class_to_label_id={
            "Annual Crop": 0,
            "Forest": 1,
            "Herbaceous Vegetation": 2,
            "Highway": 3,
            "Industrial Buildings": 4,
            "Pasture": 5,
            "Permanent Crop": 6,
            "Residential Buildings": 7,
            "River": 8,
            "SeaLake": 9,
        }
    )
    meta = ModelMetadata(
        name="eurosat",
        class_map=class_map,
        signatures=model_sig,
        artifact=model_artifact,
        ml_model_processor_type="cpu",
    )
    json_str = meta.model_dump_json(indent=2)
    with open("example.json", "w") as file:
        file.write(json_str)
    print(meta)

if __name__ == "__main__":
    app()
