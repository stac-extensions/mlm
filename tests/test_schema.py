import copy
import os
from typing import Any, cast

import pystac
import pytest
from jsonschema.exceptions import ValidationError
from pystac.validation.stac_validator import STACValidator

from stac_model.base import JSON
from stac_model.schema import SCHEMA_URI, MLModelProperties

from conftest import get_all_stac_item_examples

# ignore typing errors introduced by generic JSON manipulation errors
# mypy: disable_error_code="arg-type,call-overload,index,union-attr"


@pytest.mark.parametrize(
    "mlm_example",  # value passed to 'mlm_example' fixture
    get_all_stac_item_examples(),
    indirect=True,
)
def test_mlm_schema(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(cast(dict[str, Any], mlm_example))
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(validated) >= len(mlm_item.stac_extensions)  # extra STAC core schemas
    assert SCHEMA_URI in validated


@pytest.mark.parametrize(
    "mlm_example",
    ["item_raster_bands.json"],
    indirect=True,
)
def test_mlm_no_undefined_prefixed_field_item_properties(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    # undefined property anywhere in the schema
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["properties"]["mlm:unknown"] = "random"
    with pytest.raises(pystac.errors.STACValidationError) as exc:
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert all(
        info in str(exc.value.source)
        for info in ["mlm:unknown", "^(?!mlm:)"]
    )

    # defined property only allowed at the Asset level
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["properties"]["mlm:artifact_type"] = "torch.save"
    with pytest.raises(pystac.errors.STACValidationError) as exc:
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    errors = cast(list[ValidationError], exc.value.source)
    assert "mlm:artifact_type" in str(errors[0].validator_value)
    assert errors[0].schema["description"] == "Fields that are disallowed under the Item properties."


@pytest.mark.parametrize(
    "mlm_example",
    ["item_raster_bands.json"],
    indirect=True,
)
@pytest.mark.parametrize(
    ["test_field", "test_value"],
    [
        ("mlm:unknown", "random"),
        ("mlm:name", "test-model"),
        ("mlm:input", []),
        ("mlm:output", []),
        ("mlm:hyperparameters", {"test": {}}),
    ]
)
def test_mlm_no_undefined_prefixed_field_asset_properties(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
    test_field: str,
    test_value: Any,
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid
    assert mlm_data["assets"]["weights"]

    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["assets"]["weights"][test_field] = test_value
    with pytest.raises(pystac.errors.STACValidationError) as exc:
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(exc.value.source) == 1
    errors = cast(list[ValidationError], exc.value.source)
    assert test_field in errors[0].instance
    assert errors[0].schema["description"] in [
        "All possible MLM fields regardless of the level they apply (Collection, Item, Asset, Link).",
        "Fields that are disallowed under the Asset properties."
    ]


@pytest.mark.parametrize(
    "mlm_example",
    ["item_raster_bands.json"],
    indirect=True,
)
def test_mlm_allowed_field_asset_properties_override(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    # defined property allowed both at the Item at the Asset level
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["assets"]["weights"]["mlm:accelerator"] = "cuda"
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_raster_bands.json"],
    indirect=True,
)
def test_mlm_missing_bands_invalid_if_mlm_input_lists_bands(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_bands_bad_data = copy.deepcopy(mlm_example)
    mlm_bands_bad_data["assets"]["weights"].pop("raster:bands")  # no 'None' to raise in case missing
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
    mlm_example: dict[str, JSON],
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_eo_bands_bad_data = copy.deepcopy(mlm_example)
    mlm_eo_bands_bad_data["assets"]["weights"].pop("eo:bands")  # no 'None' to raise in case missing
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
    mlm_example: dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["properties"]["mlm:input"] = []
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)

    with pytest.raises(pystac.errors.STACValidationError):
        mlm_data["properties"].pop("mlm:input")  # no 'None' to raise in case missing
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
@pytest.mark.parametrize(
    ["test_scaling", "is_valid"],
    [
        ([{"type": "unknown", "mean": 1, "stddev": 2}], False),
        ([{"type": "min-max", "mean": 1, "stddev": 2}], False),
        ([{"type": "z-score", "minimum": 1, "maximum": 2}], False),
        ([{"type": "min-max", "mean": 1, "stddev": 2}, {"type": "min-max", "minimum": 1, "maximum": 2}], False),
        ([{"type": "z-score", "mean": 1, "stddev": 2}, {"type": "z-score", "minimum": 1, "maximum": 2}], False),
        ([{"type": "min-max", "minimum": 1, "maximum": 2}], True),
        ([{"type": "z-score", "mean": 1, "stddev": 2, "minimum": 1, "maximum": 2}], True),  # extra must be ignored
        ([{"type": "processing"}], False),
        ([{"type": "processing", "format": "test", "expression": "test"}], True),
        (
            [
                {"type": "processing", "format": "test", "expression": "test"},
                {"type": "min-max", "minimum": 1, "maximum": 2},
            ],
            True,
        ),
    ],
)
def test_mlm_input_scaling_combination(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
    test_scaling: list[dict[str, Any]],
    is_valid: bool,
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_data["properties"]["mlm:input"][0]["value_scaling"] = test_scaling  # type: ignore
    mlm_item = pystac.Item.from_dict(mlm_data)
    if is_valid:
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    else:
        with pytest.raises(pystac.errors.STACValidationError):
            pystac.validation.validate(mlm_item, validator=mlm_validator)


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
def test_mlm_other_non_mlm_assets_allowed(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # self-check valid beforehand

    mlm_data["assets"]["sample"] = {
        "type": "image/jpeg",
        "href": "https://example.com/sample/output.jpg",
        "roles": ["preview"],
        "title": "Model Output Predictions Sample",
    }
    mlm_data["assets"]["model-cart"] = {
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
    mlm_example: dict[str, JSON],
    model_asset_extras: dict[str, Any],
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
        "model": mlm_model  # type: ignore
    }
    mlm_item = pystac.Item.from_dict(mlm_data)
    if is_valid:
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    else:
        with pytest.raises(pystac.errors.STACValidationError) as exc:
            pystac.validation.validate(mlm_item, validator=mlm_validator)
        errors = cast(list[ValidationError], exc.value.source)
        assert errors[0].schema["$comment"] in [
            "At least one Asset must provide the model definition indicated by the 'mlm:model' role.",
            "Used to check the artifact type property that is required by a Model Asset annotated by 'mlm:model' role."
        ]


@pytest.mark.parametrize(
    "mlm_example",
    ["item_basic.json"],
    indirect=True,
)
def test_mlm_asset_artifact_type_checked(
    mlm_validator: STACValidator,
    mlm_example: dict[str, JSON],
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # self-check valid beforehand

    mlm_data["assets"]["model"]["mlm:artifact_type"] = 1234  # type: ignore
    mlm_item = pystac.Item.from_dict(mlm_data)
    with pytest.raises(pystac.errors.STACValidationError) as exc:
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert "1234 is not of type 'string'" in str(exc.value.source)

    mlm_data["assets"]["model"]["mlm:artifact_type"] = ""  # type: ignore
    mlm_item = pystac.Item.from_dict(mlm_data)
    with pytest.raises(pystac.errors.STACValidationError) as exc:
        pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert "should be non-empty" in str(exc.value.source)


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
    col_links: list[dict[str, str]] = mlm_example["links"]
    col_items = {os.path.basename(link["href"]) for link in col_links if link["rel"] == "item"}
    all_items = {os.path.basename(path) for path in get_all_stac_item_examples()}
    assert all_items == col_items, "Missing STAC Item examples in the example STAC Collection links."


@pytest.mark.parametrize(
    "mlm_example",
    ["mlm-metadata.yaml"],
    indirect=True,
)
def test_mlm_metadata_only_yaml_validation(mlm_example):
    MLModelProperties.model_validate(mlm_example)
