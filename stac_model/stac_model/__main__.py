import typer
from rich.console import Console

from stac_model import __version__
from stac_model.schema import InputArray, Statistics, ModelInput, Architecture, Runtime, Asset, ResultArray, ModelOutput, MLModel


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

    input_array = InputArray(
        shape=[-1, 13, 64, 64], dim_order="bchw", dtype="float32"
    )
    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    stats = Statistics(mean=[1354.40546513, 1118.24399958, 1042.92983953, 947.62620298, 1199.47283961,
                            1999.79090914, 2369.22292565, 2296.82608323, 732.08340178, 12.11327804,
                            1819.01027855, 1118.92391149, 2594.14080798],
                        stddev= [245.71762908, 333.00778264, 395.09249139, 593.75055589, 566.4170017,
                            861.18399006, 1086.63139075, 1117.98170791, 404.91978886, 4.77584468,
                            1002.58768311, 761.30323499, 1231.58581042])
    mlm_input = ModelInput(name= "13 Band Sentinel-2 Batch", bands=band_names, input_array=input_array, norm_by_channel=True, norm_type="z_score", rescale_type="none", statistics=stats, pre_processing_function = "https://github.com/microsoft/torchgeo/blob/545abe8326efc2848feae69d0212a15faba3eb00/torchgeo/datamodules/eurosat.py")
    mlm_architecture = Architecture(name = "ResNet-18", file_size=1, memory_size=1, summary= "Sourced from torchgeo python library, identifier is ResNet18_Weights.SENTINEL2_ALL_MOCO", pretrained_source="EuroSat Sentinel-2", total_parameters= 11_700_000)
    mlm_runtime = Runtime(framework= "torch", version= "2.1.2+cu121", asset= Asset(href= "https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco/resolve/main/resnet18_sentinel2_all_moco-59bfdff9.pth"),
                          source_code= Asset(href="https://github.com/microsoft/torchgeo/blob/61efd2e2c4df7ebe3bd03002ebbaeaa3cfe9885a/torchgeo/models/resnet.py#L207"), accelerator="cuda", accelerator_constrained=False, hardware_summary="Unknown")
    result_array = ResultArray(shape=[-1, 10], dim_names=["batch", "class"], dtype="float32")
    mlm_output = ModelOutput(task= "classification", number_of_classes= 10, output_shape=[-1, 10], result_array=result_array, class_name_mapping= {
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
        })
    ml_model_meta = MLModel(mlm_name="Resnet-18 Sentinel-2 ALL MOCO", mlm_input=[mlm_input], mlm_architecture=mlm_architecture, mlm_runtime=mlm_runtime, mlm_output=mlm_output)
    json_str = ml_model_meta.model_dump_json(indent=2, exclude_none=True)
    with open("example.json", "w") as file:
        file.write(json_str)
    print(ml_model_meta.model_dump_json(indent=2, exclude_none=True))
if __name__ == "__main__":
    app()
