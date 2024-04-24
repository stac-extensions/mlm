import copy
from typing import Any, Dict, cast

import pystac
import pytest
from pystac.validation.stac_validator import STACValidator

from stac_model.base import JSON
from stac_model.schema import SCHEMA_URI


@pytest.mark.parametrize(
    "mlm_example",  # value passed to 'mlm_example' fixture
    [
        "item_basic.json",
        "item_raster_bands.json",
        "item_eo_bands.json",
        "item_eo_bands_summarized.json",
        "item_eo_and_raster_bands.json",
        "item_multi_io.json",
    ],
    indirect=True,
)
def test_mlm_schema(
    mlm_validator: STACValidator,
    mlm_example: JSON,
) -> None:
    mlm_item = pystac.Item.from_dict(cast(Dict[str, Any], mlm_example))
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(validated) >= len(mlm_item.stac_extensions)  # extra STAC core schemas
    assert SCHEMA_URI in validated


@pytest.mark.parametrize(
    "mlm_example",
    ["item_eo_bands_summarized.json"],
    indirect=True,
)
def test_mlm_eo_bands_invalid_only_in_item_properties(
    mlm_validator: STACValidator,
    mlm_example: JSON,
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    pystac.validation.validate(mlm_item, validator=mlm_validator)  # ensure original is valid

    mlm_eo_bands_bad_data: Dict[str, JSON] = copy.deepcopy(mlm_example)
    mlm_eo_bands_bad_data["assets"]["weights"].pop("eo:bands")
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
    mlm_example: JSON,
) -> None:
    mlm_data = copy.deepcopy(mlm_example)
    mlm_data["properties"]["mlm:input"] = []
    mlm_item = pystac.Item.from_dict(mlm_data)
    pystac.validation.validate(mlm_item, validator=mlm_validator)

    with pytest.raises(pystac.errors.STACValidationError):
        mlm_data["properties"].pop("mlm:input")
        mlm_item = pystac.Item.from_dict(mlm_data)
        pystac.validation.validate(mlm_item, validator=mlm_validator)


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())


def test_validate_model_against_schema(eurosat_resnet, mlm_validator):
    mlm_item = pystac.read_dict(eurosat_resnet.item.to_dict())
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert SCHEMA_URI in validated
