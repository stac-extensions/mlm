from typing import Any, Optional, cast

import torch
import torch.nn as nn
from pystac import Asset, Collection, Item, Link
from pystac.extensions.eo import Band, EOExtension

from stac_model.base import TaskEnum
from stac_model.input import InputStructure, ModelInput
from stac_model.output import MLMClassification, ModelOutput, ModelResult
from stac_model.schema import ItemMLModelExtension, MLModelExtension, MLModelProperties


def normalize_dtype(torch_dtype: torch.dtype) -> str:
    """
    Convert a PyTorch dtype (e.g., torch.float32) to a standardized string
    used in metadata or schemas (e.g., 'float32').
    Raise a ValueError if the dtype is not supported.
    """
    dtype_mapping = {
        torch.uint8: "uint8",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.cfloat: "cfloat32",
        torch.cdouble: "cfloat64",
    }
    if torch_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported dtype: {torch_dtype}. Supported dtypes are: {list(dtype_mapping.keys())}")
    return dtype_mapping[torch_dtype]


def get_input_dtype(state_dict: dict) -> str:
    """
    Get the data type (dtype) of the input from the first convolutional layer's weights.
    """
    for key, tensor in state_dict.items():
        if "encoder._conv_stem.weight" in key:
            return normalize_dtype(tensor.dtype)
    raise ValueError("Could not determine input dtype from model weights.")


def get_output_dtype(state_dict: dict) -> str:
    """
    Get the data type (dtype) of the output from the segmentation head's last conv layer.
    """
    for key, tensor in reversed(state_dict.items()):
        if "segmentation_head.0.weight" in key:
            return normalize_dtype(tensor.dtype)
    raise ValueError("Could not determine output dtype from model weights.")


def get_input_channels(state_dict: dict) -> int:
    """
    Get number of input channels from the first convolutional layer's weights.
    """
    for key, tensor in state_dict.items():
        if "encoder._conv_stem.weight" in key:
            return tensor.shape[1]
    raise ValueError("Could not determine input channels from model weights.")


def get_output_channels(state_dict: dict) -> int:
    """
    Get number of output channels from the segmentation head's last conv layer.
    """
    for key, tensor in reversed(state_dict.items()):
        if "segmentation_head.0.weight" in key:
            return tensor.shape[0]
    raise ValueError("Could not determine output channels from model weights.")


def from_torch(
    model: nn.Module,
    *,
    weights: Optional[object] = None,
    item_id: str = None,
    collection_id: str | Collection | None = None,
    bbox: list[float] | None,
    geometry: dict[str, Any] | None,
    links: Optional[list[dict]] = None,
    datetime_range: tuple[str, str] = None,
    task: TaskEnum = None,
) -> ItemMLModelExtension:
    if datetime_range is None:
        raise ValueError("datetime_range must be provided as a tuple of (start, end) strings for a valid STAC item.")

    if item_id is None:
        raise ValueError("item_id must be provided as string for a valid STAC item.")

    if collection_id is None:
        raise ValueError("collection_id must be provided as string or Collection for a valid STAC item.")

    total_params = sum(p.numel() for p in model.parameters())
    module = model.__class__.__module__
    class_name = model.__class__.__name__
    arch = f"{module}.{class_name}"

    # Extra metadata only found in weights of torchgeo models
    has_meta = weights is not None and hasattr(weights, "meta")
    state_dict = model.state_dict()

    if has_meta:
        in_chans = weights.meta.get("in_chans", get_input_channels(state_dict))
        num_classes = weights.meta.get("num_classes", get_output_channels(state_dict))
    else:
        in_chans = get_input_channels(state_dict)
        num_classes = get_output_channels(state_dict)

    input_shape = [1, in_chans, 224, 224]
    output_shape = [1, num_classes]
    input_data_type = get_input_dtype(state_dict)
    output_data_type = get_output_dtype(state_dict)

    input_struct = InputStructure(
        shape=input_shape,
        dim_order=["batch", "channel", "height", "width"],
        data_type=input_data_type,
    )

    if has_meta and "bands" in weights.meta:
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

    if has_meta:
        meta = weights.meta
        url = weights.url

    mlm_props = MLModelProperties(
        name=(f"{meta.get('model', 'Model')}_{meta.get('encoder', '')}"),
        architecture=arch,
        framework=module,
        tasks=task,
        input=[model_input],
        output=[model_output],
        total_parameters=total_params,
        pretrained=True,
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

    # Publication asset
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
        datetime=None,
        properties={
            "start_datetime": datetime_range[0],
            "end_datetime": datetime_range[1],
            "description": "An Item with Machine Learning Model Extension metadata for a PyTorch model.",
        },
        stac_extensions=[MLModelExtension.get_schema_uri()],
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
