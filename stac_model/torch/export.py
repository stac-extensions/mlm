from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Protocol, Union, cast


import torch.nn as nn
from pystac import Asset, Collection, Item, Link, utils
from pystac.extensions.eo import Band, EOExtension
from shapely import geometry as geom

from stac_model.base import ModelTask
from stac_model.torch.utils import (
    extract_value_scaling,
    get_input_channels,
    get_input_dtype,
    get_input_hw,
    get_output_channels,
    get_output_dtype,
)
from stac_model.input import InputStructure, ModelInput
from stac_model.output import MLMClassification, ModelOutput, ModelResult
from stac_model.schema import ItemMLModelExtension, MLModelExtension, MLModelProperties


class WeightsWithMeta(Protocol):
    """
    Protocol for objects following the structure of torchvision.models._api.Weights.

    This is used to type hint weights objects that mimic the TorchVision Weights API,
    including torchgeo models with metadata like `meta`, `url`, and `transforms`.

    See: https://github.com/pytorch/vision/blob/main/torchvision/models/_api.py
    """

    url: str
    transforms: Callable[..., Any]
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
    links: Optional[list[dict[str, Any]]] = None,
    datetime: Optional[datetime] = None,
    datetime_range: Optional[tuple[Union[str, datetime], Union[str, datetime]]] = None,
    stac_extensions: Optional[list[str]] = None,
    stac_properties: Optional[dict[str, Any]] = None,
    # torch parameters
    weights: Optional[WeightsWithMeta] = None,
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

    bands = weights.meta["bands"] if weights and "bands" in weights.meta else None
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
