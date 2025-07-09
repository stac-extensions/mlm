import glob
import json
import os
from typing import TYPE_CHECKING, Any, cast

import pystac
import pytest
import yaml

from stac_model.base import JSON
from stac_model.examples import eurosat_resnet as make_eurosat_resnet
from stac_model.schema import SCHEMA_URI

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest

TEST_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.abspath(os.path.join(TEST_DIR, "../examples"))
JSON_SCHEMA_DIR = os.path.abspath(os.path.join(TEST_DIR, "../json-schema"))


def get_all_stac_item_examples() -> list[str]:
    all_json = glob.glob("**/*.json", root_dir=EXAMPLES_DIR, recursive=True)
    all_geojson = glob.glob("**/*.geojson", root_dir=EXAMPLES_DIR, recursive=True)
    all_stac_items = [
        path
        for path in all_json + all_geojson
        if os.path.splitext(os.path.basename(path))[0] not in ["collection", "catalog"]
    ]
    return all_stac_items


@pytest.fixture(scope="session")
def mlm_schema() -> JSON:
    with open(os.path.join(JSON_SCHEMA_DIR, "schema.json")) as schema_file:
        data = json.load(schema_file)
    return cast(JSON, data)


@pytest.fixture(scope="session")
def mlm_validator(
    request: "SubRequest",
    mlm_schema: dict[str, Any],
) -> pystac.validation.stac_validator.JsonSchemaSTACValidator:
    """
    Update the :class:`pystac.validation.RegisteredValidator` with the local MLM JSON schema definition.

    Because the schema is *not yet* uploaded to the expected STAC schema URI,
    any call to :func:`pystac.validation.validate` or :meth:`pystac.stac_object.STACObject.validate` results
    in ``GetSchemaError`` when the schema retrieval is attempted by the validator.By adding the schema to the
    mapping beforehand, remote resolution can be bypassed temporarily. When evaluating modifications to the
    current schema, this also ensures that local changes are used instead of the remote reference.
    """
    validator = pystac.validation.RegisteredValidator.get_validator()
    validator = cast(pystac.validation.stac_validator.JsonSchemaSTACValidator, validator)
    validator.schema_cache[SCHEMA_URI] = mlm_schema
    pystac.validation.RegisteredValidator.set_validator(validator)  # apply globally to allow 'STACObject.validate()'
    return validator


@pytest.fixture
def mlm_example(request: "SubRequest") -> dict[str, JSON]:
    with open(os.path.join(EXAMPLES_DIR, request.param)) as example_file:
        if request.param.endswith(".json"):
            data = json.load(example_file)
        elif request.param.endswith(".yaml"):
            data = yaml.safe_load(example_file)
        else:
            raise ValueError(f"Unsupported file format for example: {request.param}")
    return cast(dict[str, JSON], data)


@pytest.fixture(name="eurosat_resnet")
def eurosat_resnet():
    return make_eurosat_resnet()
