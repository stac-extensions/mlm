import json
import os
from typing import Any, Dict, cast

import pystac
import pytest

from stac_model.examples import eurosat_resnet as make_eurosat_resnet
from stac_model.schema import SCHEMA_URI

TEST_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.abspath(os.path.join(TEST_DIR, "../examples"))
JSON_SCHEMA_DIR = os.path.abspath(os.path.join(TEST_DIR, "../json-schema"))


@pytest.fixture(scope="session")
def mlm_schema() -> Dict[str, Any]:
    with open(os.path.join(JSON_SCHEMA_DIR, "schema.json")) as schema_file:
        return json.load(schema_file)


@pytest.fixture(scope="session", autouse=True)
def mlm_validator(
    request: pytest.FixtureRequest,
    mlm_schema: Dict[str, Any],
) -> pystac.validation.stac_validator.JsonSchemaSTACValidator:
    """
    Update the :class:`pystac.validation.RegisteredValidator` with the local ML-AOI JSON schema definition.

    Because the schema is *not yet* uploaded to the expected STAC schema URI,
    any call to :func:`pystac.validation.validate` or :meth:`pystac.stac_object.STACObject.validate` results
    in ``GetSchemaError`` when the schema retrieval is attempted by the validator. By adding the schema to the
    mapping beforehand, remote resolution can be bypassed temporarily.
    """
    validator = pystac.validation.RegisteredValidator.get_validator()
    validator = cast(pystac.validation.stac_validator.JsonSchemaSTACValidator, validator)
    validator.schema_cache[SCHEMA_URI] = mlm_schema
    pystac.validation.RegisteredValidator.set_validator(validator)  # apply globally to allow 'STACObject.validate()'
    return validator


@pytest.fixture(scope="session", autouse=True)
def mlm_example() -> Dict[str, Any]:
    with open(os.path.join(EXAMPLES_DIR, "example.json")) as example_file:
        return json.load(example_file)


@pytest.fixture(name="eurosat_resnet")
def eurosat_resnet():
    return make_eurosat_resnet()
