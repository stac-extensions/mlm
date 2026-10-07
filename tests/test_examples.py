import copy
from typing import Any

import pystac
import pytest

from stac_model.schema import SCHEMA_URI

from conftest import get_all_stac_item_mlm_examples

# ignore typing errors introduced by generic JSON manipulation errors
# mypy: disable_error_code="arg-type,call-overload,index,union-attr"


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())


def test_validate_model_against_schema(eurosat_resnet, mlm_validator):
    mlm_item = pystac.read_dict(eurosat_resnet.item.to_dict())
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert SCHEMA_URI in validated


def _assert_model_reference_names(item: dict[str, Any]) -> None:
    """
    Check direct input/output names against Item and Asset band/variable definitions.
    """
    properties = item["properties"]
    definitions = [properties, *item.get("assets", {}).values()]
    names: dict[str, set[str]] = {"bands": set(), "variables": set()}
    for definition in definitions:
        names["variables"].update(definition.get("cube:variables", {}))
        for field in ("bands", "eo:bands", "raster:bands"):
            for band in definition.get(field, []):
                name = band.get("name", band.get("common_name", band.get("eo:common_name")))
                if name is not None:
                    names["bands"].add(name)

    for direction in ("input", "output"):
        for index, model_io in enumerate(properties.get(f"mlm:{direction}", [])):
            for field in ("bands", "variables"):
                for reference in model_io.get(field, []):
                    name = reference if isinstance(reference, str) else reference["name"]
                    if isinstance(reference, dict) and "format" in reference and "expression" in reference:
                        # An expression can define a virtual band/variable absent from the raw-data definitions.
                        continue
                    assert name in names[field], (
                        f"{item['id']}: mlm:{direction}[{index}] ({model_io['name']!r}) "
                        f"{field} reference {name!r} has no matching Item or Asset definition. "
                        f"Available names: {sorted(names[field])}"
                    )


@pytest.mark.parametrize("mlm_example", get_all_stac_item_mlm_examples(), indirect=True)
def test_example_input_output_reference_names(mlm_example: dict[str, Any]) -> None:
    """
    Tests that direct band/variable references resolve across every example's inputs and outputs.

    Expression-bearing objects can define virtual channels. Empty or absent reference arrays require
    no matching definitions, and referenced channels need not exhaust all defined bands or variables.
    """
    _assert_model_reference_names(mlm_example)


@pytest.mark.parametrize(
    ("mlm_example", "field"),
    [
        ("item_eo_bands.json", "bands"),
        ("item_datacube_variables.json", "variables"),
    ],
    indirect=["mlm_example"],
)
@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("object_reference", [False, True], ids=["string", "object"])
def test_example_reference_name_mismatch(
    mlm_example: dict[str, Any], field: str, direction: str, object_reference: bool
) -> None:
    """
    Tests rejection of undefined string/object references for either model interface and reference kind.
    """
    item = copy.deepcopy(mlm_example)
    model_io = item["properties"][f"mlm:{direction}"][0]
    model_io[field] = [{"name": "undefined-reference"}] if object_reference else ["undefined-reference"]
    with pytest.raises(AssertionError, match=rf"mlm:{direction}\[0\].*{field} reference 'undefined-reference'"):
        _assert_model_reference_names(item)


@pytest.mark.parametrize("mlm_example", ["item_datacube_variables.json"], indirect=True)
def test_example_temperature_variable_definition_mismatch(mlm_example: dict[str, Any]) -> None:
    """
    Tests detection of the original Datacube temperature-key mismatch without changing its references.
    """
    item = copy.deepcopy(mlm_example)
    variables = item["properties"]["cube:variables"]
    variables["2m_temperature"] = variables.pop("temperature_2m")
    with pytest.raises(AssertionError, match="variables reference 'temperature_2m'"):
        _assert_model_reference_names(item)


@pytest.mark.parametrize("mlm_example", ["item_eo_bands.json"], indirect=True)
@pytest.mark.parametrize("band_field", ["bands", "eo:bands", "raster:bands"])
@pytest.mark.parametrize("location", ["properties", "asset"])
def test_example_band_definition_locations(mlm_example: dict[str, Any], band_field: str, location: str) -> None:
    """
    Tests direct object references against STAC common, EO, or Raster bands on Items and Assets.
    """
    item = copy.deepcopy(mlm_example)
    assets = item["assets"]
    bands = copy.deepcopy(assets["weights"].pop("eo:bands"))
    definition = item["properties"] if location == "properties" else assets["weights"]
    definition[band_field] = bands
    for reference in item["properties"]["mlm:input"]:
        reference["bands"] = [{"name": name} for name in reference["bands"]]
    _assert_model_reference_names(item)


@pytest.mark.parametrize("mlm_example", ["item_datacube_variables.json"], indirect=True)
def test_example_asset_variable_definitions(mlm_example: dict[str, Any]) -> None:
    """
    Tests variable objects referencing Datacube definitions on a model Asset rather than the Item.
    """
    item = copy.deepcopy(mlm_example)
    properties = item["properties"]
    asset = next(iter(item["assets"].values()))
    asset["cube:variables"] = properties.pop("cube:variables")
    for direction in ("input", "output"):
        for model_io in properties[f"mlm:{direction}"]:
            model_io["variables"] = [{"name": name} for name in model_io["variables"]]
    _assert_model_reference_names(item)


@pytest.mark.parametrize("mlm_example", ["item_bands_expression.json"], indirect=True)
@pytest.mark.parametrize("missing_field", ["format", "expression"])
def test_example_incomplete_virtual_band_reference(mlm_example: dict[str, Any], missing_field: str) -> None:
    """
    Tests that an incomplete expression object cannot exempt an undefined band from name resolution.
    """
    item = copy.deepcopy(mlm_example)
    derived_band = item["properties"]["mlm:input"][0]["bands"][-1]
    derived_band.pop(missing_field)
    with pytest.raises(AssertionError, match="bands reference 'NDVI'"):
        _assert_model_reference_names(item)
