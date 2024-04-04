from typing import Any, Dict
import pystac
import pytest

from stac_model.schema import SCHEMA_URI


@pytest.mark.parametrize(
    "mlm_example",  # value passed to 'mlm_example' fixture
    [
        "example.json",
        "example_eo_bands.json",
    ],
    indirect=True,
)
def test_mlm_schema(
    mlm_validator: pystac.validation.STACValidator,
    mlm_example,
) -> None:
    mlm_item = pystac.Item.from_dict(mlm_example)
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(validated) >= len(mlm_item.stac_extensions)  # extra STAC core schemas
    assert SCHEMA_URI in validated


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())
