import json
import os
from typing import TYPE_CHECKING, Any, Dict, cast

import pystac
import pytest

from stac_model.base import JSON
from stac_model.examples import eurosat_resnet as make_eurosat_resnet
from stac_model.schema import SCHEMA_URI

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest

TEST_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.abspath(os.path.join(TEST_DIR, "../examples"))
JSON_SCHEMA_DIR = os.path.abspath(os.path.join(TEST_DIR, "../json-schema"))


@pytest.fixture(scope="session")
def mlm_schema() -> JSON:
    with open(os.path.join(JSON_SCHEMA_DIR, "schema.json")) as schema_file:
        data = json.load(schema_file)
    return cast(JSON, data)


@pytest.fixture(scope="session")
def mlm_validator(
    request: "SubRequest",
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


@pytest.fixture
def mlm_example(request: "SubRequest") -> JSON:
    with open(os.path.join(EXAMPLES_DIR, request.param)) as example_file:
        data = json.load(example_file)
    return cast(JSON, data)


@pytest.fixture(name="eurosat_resnet")
def eurosat_resnet():
    return make_eurosat_resnet()
