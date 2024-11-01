import copy
import os
from typing import Any, Dict, List, cast

import pystac
import pytest
from pystac.validation.stac_validator import STACValidator

from stac_model.base import JSON
from stac_model.schema import SCHEMA_URI

from conftest import get_all_stac_item_examples


@pytest.mark.parametrize(
    "mlm_example",  # value passed to 'mlm_example' fixture
    get_all_stac_item_examples(),
    indirect=True,
)
def test_mlm_schema(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(cast(Dict[str, Any], mlm_example))
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(validated) >= len(mlm_item.stac_extensions)  # extra STAC core schemas
    assert SCHEMA_URI in validated


@pytest.mark.parametrize(
    "mlm_example",
    ["item_raster_bands.json"],
    indirect=True,
)
def test_mlm_missing_bands_invalid_if_mlm_input_lists_bands(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_bands_bad_data = copy.deepcopy(mlm_example)
    mlm_bands_bad_data["assets"]["weights"].pop("raster:bands")  # type: ignore  # no 'None' to raise in case modified
    with pytest.raises(pystac.errors.STACValidationError):
        mlm_bands_bad_item = pystac.Item.from_dict(mlm_bands_bad_data)
        pystac.validation.validate(mlm_bands_bad_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_eo_bands_summarized.json"],
    indirect=True,
)
def test_mlm_eo_bands_invalid_only_in_item_properties(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_eo_bands_bad_data = copy.deepcopy(mlm_example)
    mlm_eo_bands_bad_data["assets"]["weights"].pop("eo:bands")  # type: ignore  # no 'None' to raise in case modified
    with pytest.raises(pystac.errors.STACValidationError):
        mlm_eo_bands_bad_item = pystac.Item.from_dict(mlm_eo_bands_bad_data)
        pystac.validation.validate(mlm_eo_bands_bad_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
def test_mlm_no_input_allowed_but_explicit_empty_array_required(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["properties"]["mlm:input"] = []  # type: ignore
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)

    with pytest.raises(pystac.errors.STACValidationError):
        mlm_data["properties"].pop("mlm:input")  # type: ignore  # no 'None' to raise in case modified
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
def test_mlm_other_non_mlm_assets_allowed(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # self-check valid beforehand

    mlm_data["assets"]["sample"] = {  # type: ignore
        "type": "image/jpeg",
        "href": "https://example.com/sample/output.jpg",
        "roles": ["preview"],
        "title": "Model Output Predictions Sample",
    }
    mlm_data["assets"]["model-cart"] = {  # type: ignore
        "type": "text/markdown",
        "href": "https://example.com/sample/model.md",
        "roles": ["metadata"],
        "title": "Model Cart",
    }
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # still valid


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
@pytest.mark.parametrize(
    ["model_asset_extras", "is_valid"],
    [
        ({"roles": ["checkpoint"]}, False),
        ({"roles": ["checkpoint", "mlm:model"]}, False),
        ({"roles": ["checkpoint"], "mlm:artifact_type": "test"}, False),
        ({"roles": ["checkpoint", "mlm:model"], "mlm:artifact_type": "test"}, True),
    ]
)
def test_mlm_at_least_one_asset_model(
    mlm_validator: STACValidator,
    mlm_example: Dict[str, JSON],
    model_asset_extras: Dict[str, Any],
    is_valid: bool,
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # self-check valid beforehand

    mlm_model = {
        "type": "application/octet-stream; application=pytorch",
        "href": "https://example.com/sample/checkpoint.pt",
        "title": "Model Weights Checkpoint",
    }
    mlm_model.update(model_asset_extras)
    mlm_data["assets"] = {
        "model": mlm_model
    }
    mlm_item = pystac.Item.from_dict(mlm_data)
    if is_valid:
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    else:
        with pytest.raises(pystac.errors.STACValidationError) as exc:
            pystac.validation.validate(mlm_item, validator=mlm_validator)
        assert exc.value.source[0].schema["$comment"] in [
            "At least one Asset must provide the model definition indicated by the 'mlm:model' role.",
            "Used to check the artifact type property that is required by a Model Asset annotated by 'mlm:model' role."
        ]


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())


def test_validate_model_against_schema(eurosat_resnet, mlm_validator):
    mlm_item = pystac.read_dict(eurosat_resnet.item.to_dict())
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert SCHEMA_URI in validated


@pytest.mark.parametrize(
    "mlm_example",
    ["collection.json"],
    indirect=True,
)
def test_collection_include_all_items(mlm_example):
    """
    This is only for self-validation, to make sure all examples are contained in the example STAC collection.
    """
    col_links: List[Dict[str, str]] = mlm_example["links"]
    col_items = {os.path.basename(link["href"]) for link in col_links if link["rel"] == "item"}
    all_items = {os.path.basename(path) for path in get_all_stac_item_examples()}
    assert all_items == col_items, "Missing STAC Item examples in the example STAC Collection links."
