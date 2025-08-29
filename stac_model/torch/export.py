import logging
import pathlib
import tempfile
from datetime import datetime
from typing import Any, Protocol, Union, cast

import torch
import torch.nn as nn
from kornia.augmentation import AugmentationSequential
from pystac import Asset, Collection, Item, Link, utils
from pystac.extensions.eo import Band, EOExtension
from shapely import geometry as geom
from torch.export.dynamic_shapes import Dim
from torch.export.pt2_archive._package import package_pt2

from ..base import ModelTask, Path
from ..input import InputStructure, ModelInput
from ..output import MLMClassification, ModelOutput, ModelResult
from ..schema import ItemMLModelExtension, MLModelExtension, MLModelProperties
from .base import AOTIFiles, ExportedPrograms, ExtraFiles
from .utils import (
    aoti_compile,
    create_example_input_from_shape,
    extract_module_arg_names,
    extract_value_scaling,
    get_input_channels,
    get_input_dtype,
    get_input_hw,
    get_output_channels,
    get_output_dtype,
    model_properties_to_metadata,
    update_properties,
)

logger = logging.getLogger(__name__)


class WeightsWithMeta(Protocol):
    """
    Protocol for objects following the structure of torchvision.models._api.Weights.

    This is used to type hint weights objects that mimic the TorchVision Weights API,
    including torchgeo models with metadata like `meta`, `url`, and `transforms`.

    See: https://github.com/pytorch/vision/blob/main/torchvision/models/_api.py
    """

    url: str
    transforms: AugmentationSequential
    meta: dict[str, Any]


def from_torch(
    model: nn.Module,
    task: set[ModelTask],
    *,
    # STAC Item parameters
    item_id: str,
    collection: str | Collection,
    bbox: list[float] | None,
    geometry: dict[str, Any] | None,
    links: list[dict[str, Any]] | None = None,
    datetime: datetime | None = None,
    datetime_range: tuple[Union[str, datetime], Union[str, datetime]] | None = None,
    stac_extensions: list[str] | None = None,
    stac_properties: dict[str, Any] | None = None,
    # torch parameters
    weights: WeightsWithMeta | None = None,
) -> ItemMLModelExtension:
    """Creates a STAC Item with ML Model Extension metadata from a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to export.
        task (set[ModelTask]): Set of ML tasks the model supports (e.g., classification, segmentation).
        item_id (str): Unique identifier for the STAC item.
        collection (str | Collection): Collection ID or Collection object the item belongs to.
        bbox (list[float] | None): Bounding box of the item.
        geometry (dict[str, Any] | None): GeoJSON-like geometry defining the spatial extent of the item.
        links (Optional[list[dict[str, Any]]], optional): List of STAC links to associate with the item.
        datetime (Optional[datetime], optional): Timestamp associated with the item.
        datetime_range (Optional[tuple[Union[str, datetime], Union[str, datetime]]], optional): Temporal extent as a start/end range.
        stac_extensions (Optional[list[str]], optional): Additional STAC extensions to include.
        stac_properties (Optional[dict[str, Any]], optional): Additional custom STAC properties.
        weights (Optional[WeightsWithMeta], optional): Optional PyTorch weights object with metadata.

    Returns:
        ItemMLModelExtension: A STAC Item with the Machine Learning Model extension populated.
    """
    if bbox is None and geometry is None:
        raise ValueError("Either bbox or geometry must be provided for a valid STAC item.")

    if bbox is None:
        assert geometry is not None
        bbox = utils.geometry_to_bbox(geometry)

    if geometry is None:
        geometry = geom.box(*bbox).__geo_interface__

    properties = {
        "description": "An Item with Machine Learning Model Extension metadata for a PyTorch model.",
    }

    if datetime_range:
        properties.update(
            {
                "start_datetime": str(datetime_range[0]),
                "end_datetime": str(datetime_range[1]),
            }
        )

    if not datetime and not datetime_range:
        raise ValueError("datetime or datetime range must be provided for a valid STAC item.")

    if item_id is None:
        raise ValueError("item_id must be provided as string for a valid STAC item.")

    collection_id = collection.id if isinstance(collection, Collection) else collection

    if stac_properties:
        properties.update(stac_properties)

    total_params = sum(p.numel() for p in model.parameters())
    module = model.__class__.__module__
    class_name = model.__class__.__name__
    arch = f"{module}.{class_name}"

    # Extra metadata only found in weights of torchgeo models
    has_meta = weights is not None and hasattr(weights, "meta")
    weights = cast(WeightsWithMeta, weights) if has_meta else None
    state_dict = model.state_dict()

    if weights:
        in_chans = weights.meta.get("in_chans", get_input_channels(state_dict))
        num_classes = weights.meta.get("num_classes", get_output_channels(state_dict))
    else:
        in_chans = get_input_channels(state_dict)
        num_classes = get_output_channels(state_dict)

    h, w = get_input_hw(state_dict)
    input_shape = [-1, in_chans, h, w]
    output_shape = [-1, num_classes]
    input_data_type = get_input_dtype(state_dict)
    output_data_type = get_output_dtype(state_dict)

    input_struct = InputStructure(
        shape=input_shape,
        dim_order=["batch", "bands", "height", "width"],
        data_type=input_data_type,
    )

    bands = weights.meta.get("bands", []) if weights else []
    transforms = weights.transforms if weights and hasattr(weights, "transforms") else None
    value_scaling = extract_value_scaling(transforms) if transforms else None

    model_input = ModelInput(
        name="model_input",
        bands=bands,
        input=input_struct,
        value_scaling=value_scaling,
        resize_type=None,
        pre_processing_function=None,
    )

    classes = getattr(
        model,
        "classes",
        [
            MLMClassification(value=i, name=f"class_{i}", description=f"Auto-generated class {i}")
            for i in range(output_shape[-1])
        ],
    )

    model_output = ModelOutput(
        name="model_output",
        tasks=task,
        result=ModelResult(
            shape=output_shape,
            dim_order=["batch", "classes"],
            data_type=output_data_type,
        ),
        classes=classes,
        post_processing_function=None,
    )

    # If weights are provided with metadata, extract metadata from them
    meta, url = (weights.meta, weights.url) if weights else ({}, None)
    raw_model = meta.get("model", "Model")
    model_encoder = meta.get("encoder", "Encoder")
    model_name = f"{raw_model}_{model_encoder}"
    license = meta.get("license", "license")
    publication_url = meta.get("publication", None)
    pretrained = weights is not None
    mlm_props = MLModelProperties(
        name=model_name,
        architecture=arch,
        framework=module,
        tasks=task,
        input=[model_input],
        output=[model_output],
        total_parameters=total_params,
        pretrained=pretrained,
        pretrained_source=None,
    )

    assets = {}

    # Model asset
    if url:
        assets["model"] = Asset(
            title=model_name,
            description=(f"A {raw_model} segmentation model with {model_encoder} encoder Weights are {license}."),
            href=url,
            media_type="application/octet-stream; application=pytorch",
            roles=[
                "mlm:model",
                "mlm:weights",
                "data",
            ],
            extra_fields={"mlm:artifact_type": "torch.save"},
        )

    links = links or []

    if publication_url:
        title = "Publication for the training dataset of the model"
        links.append(
            {
                "rel": "cite-as",
                "target": publication_url,
                "media_type": "text/html",
                "title": title,
            }
        )

    # Source code asset
    if url and "segmentation" in arch:
        # Define more href depending of model architecture
        assets["source_code"] = Asset(
            title=f"Source code for {model_name}",
            description="GitHub repo of the pytorch model",
            href="https://github.com/qubvel-org/segmentation_models.pytorch",
            media_type="text/html",
            roles=[
                "mlm:source_code",
                "code",
            ],
        )

    item = Item(
        id=item_id,
        collection=collection_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime,
        properties=properties,
        stac_extensions=[MLModelExtension.get_schema_uri()] + (stac_extensions or []),
        assets=assets,
        extra_fields={"mlm:entrypoint": arch},
    )

    for link in links:
        item.add_link(Link(**link))

    ext = MLModelExtension.ext(item, add_if_missing=True)
    ext.apply(mlm_props)

    if "model" in assets:
        eo_model_asset = cast(
            EOExtension[Asset],
            EOExtension.ext(assets["model"], add_if_missing=True),
        )
        eo_bands = []
        for name in bands:
            band = Band({})
            band.apply(name=name)
            eo_bands.append(band)
        eo_model_asset.apply(bands=eo_bands)

    item_mlm = MLModelExtension.ext(item, add_if_missing=True)
    item_mlm.apply(mlm_props.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=True))
    return item_mlm


@torch.no_grad()
def export(
    input_shape: list[int],
    model: torch.nn.Module,
    transforms: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram | None]:
    """Exports a model and its transforms to programs.

    Args:
        model: The model to export.
        transforms: The transforms to export. The transforms should be a `torch.nn.Module. If you have
            multiple transforms, it is recommended to wrap them in a `torch.nn.Sequential`.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        device: The device to export the model and transforms to.
        dtype: The data type to use for the model and transforms. Defaults to torch.float32.

    Returns:
        The exported model and transforms programs.
    """
    example_inputs = create_example_input_from_shape(input_shape).to(device).to(dtype)
    dims = tuple(Dim.AUTO if dim == -1 else dim for dim in input_shape)
    logger.debug("Exporting with dims: %s", dims)
    logger.debug("Example input shape: %s", list(example_inputs.shape))

    model.eval()
    model = model.to(device).to(dtype)
    model_arg = extract_module_arg_names(model)
    model_program = torch.export.export(mod=model, args=(example_inputs,), dynamic_shapes={model_arg: dims})

    if transforms is not None:
        transforms.eval()
        transforms = transforms.to(device).to(dtype)
        transforms_arg = extract_module_arg_names(transforms)
        transforms_program = torch.export.export(
            mod=transforms, args=(example_inputs,), dynamic_shapes={transforms_arg: dims}
        )
    else:
        transforms_program = None

    return model_program, transforms_program


def package(
    output_file: Path,
    model_program: torch.export.ExportedProgram,
    transforms_program: torch.export.ExportedProgram | None = None,
    metadata_properties: MLModelProperties | None = None,
    aoti_compile_and_package: bool = False,
) -> None:
    """Packages a model and its transforms AOTI exported programs into a single archive file.

    Args:
        output_file: The path to the output archive file.
        model_program: The exported model program.
        transforms_program: The exported transforms program.
        metadata_properties: MLModelProperties object
        aoti_compile_and_package: Whether to compile and package the model and transforms using AOTI.

    Raises:
        ValidationError: if the model metadata is not valid MLModelProperties.
    """
    aoti_files: AOTIFiles = {}
    extra_files: ExtraFiles = {}
    exported_programs: ExportedPrograms = {}

    if metadata_properties is not None:
        metadata_yaml = model_properties_to_metadata(metadata_properties)
        extra_files["mlm-metadata"] = metadata_yaml

    if aoti_compile_and_package:
        model_tmpdir = tempfile.TemporaryDirectory()
        transforms_tmpdir = tempfile.TemporaryDirectory()
        aoti_files.update(
            aoti_compile(
                model_directory=pathlib.Path(model_tmpdir.name),
                model_program=model_program,
                transforms_directory=pathlib.Path(transforms_tmpdir.name),
                transforms_program=transforms_program,
            )
        )
    else:
        exported_programs["model"] = model_program
        if transforms_program is not None:
            exported_programs["transforms"] = transforms_program

    package_pt2(
        f=output_file,
        exported_programs=exported_programs or None,  # type: ignore[arg-type]
        aoti_files=aoti_files or None,  # type: ignore[arg-type]
        extra_files=extra_files or None,  # type: ignore[arg-type]
    )

    if aoti_compile_and_package:
        model_tmpdir.cleanup()
        transforms_tmpdir.cleanup()


def save(
    output_file: Path,
    input_shape: list[int],
    model: torch.nn.Module,
    transforms: torch.nn.Module | None = None,
    metadata: Path | MLModelProperties | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    aoti_compile_and_package: bool = False,
) -> None:
    """Exports a model and its transforms to a packaged archive file.
    Args:
        output_file: The path to the output archive file.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        model: The model to export.
        transforms: The transforms to export. The transforms should be a `torch.nn.Module`. If you have
            multiple transforms, it is recommended to wrap them in a `torch.nn.Sequential`.
        metadata: Path to the YAML file containing model metadata or an instance of MLModelProperties.
        device: The device to export the model and transforms to.
        dtype: The data type to use for the model and transforms. Defaults to torch.float32.
        aoti_compile_and_package: Whether to compile and package the model and transforms using AOTI.
    """
    model_program, transforms_program = export(
        model=model,
        transforms=transforms,
        input_shape=input_shape,
        device=device,
        dtype=dtype,
    )

    if metadata is not None:
        metadata_properties = update_properties(metadata=metadata, input_shape=input_shape, device=device, dtype=dtype)
    else:
        metadata_properties = None

    package(
        output_file=output_file,
        model_program=model_program,
        transforms_program=transforms_program,
        metadata_properties=metadata_properties,
        aoti_compile_and_package=aoti_compile_and_package,
    )
