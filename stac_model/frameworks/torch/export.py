from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Protocol, Union, cast

from pystac import Asset, Collection, Item, Link, utils
from pystac.extensions.eo import Band, EOExtension
from shapely import geometry as geom

import torch
import torch.nn as nn
from stac_model.base import DataType, ModelTask
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


def normalize_dtype(torch_dtype: torch.dtype) -> DataType:
    """
    Convert a PyTorch dtype (e.g., torch.float32) to a standardized DataType.
    """
    return cast(DataType, str(torch_dtype).rsplit(".", 1)[-1])


def find_tensor_by_key(state_dict: dict[str, torch.Tensor], key_substring: str, reverse: bool = False) -> torch.Tensor:
    items = reversed(state_dict.items()) if reverse else state_dict.items()
    for key, tensor in items:
        if key_substring in key:
            return tensor
    raise ValueError(f"Could not find tensor with key containing '{key_substring}'")


def get_input_hw(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return tensor.shape[2], tensor.shape[3]


def get_input_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the input from the first convolutional layer's weights.
    """
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return normalize_dtype(tensor.dtype)


def get_output_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the output from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return normalize_dtype(tensor.dtype)


def get_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of input channels from the first convolutional layer's weights.
    """    
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return tensor.shape[1]


def get_output_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of output channels from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return tensor.shape[0]


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
        properties.update({
            "start_datetime": str(datetime_range[0]),
            "end_datetime": str(datetime_range[1]),
        })
    
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
    input_shape = [1, in_chans, h, w]
    output_shape = [1, num_classes]
    input_data_type = get_input_dtype(state_dict)
    output_data_type = get_output_dtype(state_dict)

    input_struct = InputStructure(
        shape=input_shape,
        dim_order=["batch", "channel", "height", "width"],
        data_type=input_data_type,
    )

    if weights and "bands" in weights.meta:
        bands = weights.meta["bands"]
    else:
        bands = [f"band_{i}" for i in range(input_shape[1])]

    model_input = ModelInput(
        name="model_input",
        bands=bands,
        input=input_struct,
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

    if weights:
        meta = weights.meta
        url = weights.url

    pretrained = weights is not None
    mlm_props = MLModelProperties(
        name=(f"{meta.get('model', 'Model')}_{meta.get('encoder', '')}"),
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

    # Model weights asset
    assets["model"] = Asset(
        title=(f"{meta.get('model', 'Model')}_{meta.get('encoder', '')}"),
        description=(
            f"A {meta.get('model', 'Model')} segmentation model with {meta.get('encoder', '')} encoder "
            f"trained on {meta.get('dataset', 'dataset')} imagery with {meta.get('num_classes', '?')}-class labels. "
            f"Weights are {meta.get('license', 'licensed')}."
        ),
        href=url,
        media_type="application/octet-stream; application=pytorch",
        roles=[
            "mlm:model",
            "mlm:weights",
            "data",
        ],
        extra_fields={"mlm:artifact_type": "torch.save"},
    )

    # Publication TODO https://github.com/stac-extensions/scientific?tab=readme-ov-file#relation-types
    # stac link rel=cite as <cite> <cite>
    # pystac link avec rel cite as <cite>

    publication_url = meta.get("publication")
    if publication_url:
        assets["publication"] = Asset(
            title=f"{meta.get('dataset', 'Dataset')} publication",
            description=f"Paper describing the {meta.get('dataset', 'dataset')} dataset and model benchmarks.",
            href=publication_url,
            media_type="text/html",
            roles=[
                "publication",
                "paper",
            ],
        )

    # Source code asset
    repo_url = meta.get("repo")
    if repo_url:
        assets["source_code"] = Asset(
            title=f"{meta.get('dataset', 'Dataset')} Baselines repository",
            description=f"GitHub repo with baseline code for {meta.get('dataset', 'dataset')} dataset models.",
            href=repo_url,
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
        stac_extensions=[MLModelExtension.get_schema_uri()]  + (stac_extensions or []),
        assets=assets,
    )

    for link in links or []:
        item.add_link(Link(**link))

    ext = MLModelExtension.ext(item, add_if_missing=True)
    ext.apply(mlm_props)

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
