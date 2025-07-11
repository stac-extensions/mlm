from typing import Optional, cast

import torch.nn as nn
from pystac import Asset, Item, Link
from pystac.extensions.eo import Band, EOExtension

from stac_model.base import TaskEnum
from stac_model.input import InputStructure, ModelInput
from stac_model.output import MLMClassification, ModelOutput, ModelResult
from stac_model.schema import ItemMLModelExtension, MLModelExtension, MLModelProperties


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
    item_id: str = "torch-model",
    bbox: Optional[list[float]] = None,
    geometry: Optional[dict] = None,
    links: Optional[list[dict]] = None,
    datetime_range: tuple[str, str] = (
        "2015-06-23T00:00:00Z",  # Sentinel-2A launch date (first Sentinel-2 data available)
        "2024-08-27T23:59:59Z",  # Dataset publication date Fields of The World (FTW)
    ),
) -> ItemMLModelExtension:
    total_params = sum(p.numel() for p in model.parameters())
    arch = f"{model.__class__.__module__}.{model.__class__.__name__}"
    task = {TaskEnum.CLASSIFICATION}

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

    input_struct = InputStructure(
        shape=input_shape,
        dim_order=["batch", "channel", "height", "width"],
        data_type="float32",
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
            data_type="float32",
        ),
        classes=classes,
        post_processing_function=None,
    )

    mlm_props = MLModelProperties(
        name=item_id,
        architecture=arch,
        tasks=task,
        input=[model_input],
        output=[model_output],
        total_parameters=total_params,
        pretrained=True,
        pretrained_source=None,
    )

    bbox = bbox or [-7.88, 37.13, 27.91, 58.21]
    geometry = geometry or {
        "type": "Polygon",
        "coordinates": [
            [
                [-7.88, 37.13],
                [-7.88, 58.21],
                [27.91, 58.21],
                [27.91, 37.13],
                [-7.88, 37.13],
            ]
        ],
    }

    if has_meta:
        meta = weights.meta
        url = weights.url
    assets = {}

    # Model weights asset
    assets["model"] = Asset(
        title=(
            f"{meta.get('model', 'Model')} ({meta.get('encoder', '')}) weights "
            f"trained on {meta.get('dataset', 'dataset')} dataset"
        ),
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
