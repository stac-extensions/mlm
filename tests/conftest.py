import functools
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


@functools.cache
def get_all_stac_mlm_examples() -> list[str]:
    all_json = glob.glob("**/*.json", root_dir=EXAMPLES_DIR, recursive=True)
    all_geojson = glob.glob("**/*.geojson", root_dir=EXAMPLES_DIR, recursive=True)
    return all_json + all_geojson


@functools.cache
def get_all_stac_collection_mlm_examples() -> list[str]:
    all_stac_collections = [
        path
        for path in get_all_stac_mlm_examples()
        if os.path.splitext(os.path.basename(path))[0] == "collection"
    ]
    return all_stac_collections


@functools.cache
def get_all_stac_item_mlm_examples() -> list[str]:
    all_stac_items = [
        path
        for path in get_all_stac_mlm_examples()
        if os.path.splitext(os.path.basename(path))[0] not in ["collection", "catalog"]
    ]
    return all_stac_items


@pytest.fixture(scope="session")
def mlm_schema() -> JSON:
    with open(os.path.join(JSON_SCHEMA_DIR, "schema.json"), mode="r", encoding="utf-8") as schema_file:
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


@functools.cache
def load_mlm_example(file_name: str) -> dict[str, JSON]:
    with open(os.path.join(EXAMPLES_DIR, file_name), mode="r", encoding="utf-8") as example_file:
        if file_name.endswith(".json"):
            data = json.load(example_file)
        elif file_name.endswith(".yaml"):
            data = yaml.safe_load(example_file)
        else:
            raise ValueError(f"Unsupported file format for example: {file_name}")
    return cast(dict[str, JSON], data)


@pytest.fixture
def mlm_example(request: "SubRequest") -> dict[str, JSON]:
    """
    Fixture that loads an example STAC Item with MLM extension from the examples directory.

    Usage:

        ```python
        @pytest.mark.parametrize(
            "mlm_example",
            ["path/to/example1.json", "path/to/example2.yaml"],  # or just the name if in 'EXAMPLES_DIR'
            indirect=True,
        )
        def test_example(mlm_example: dict[str, JSON]) -> None: ...
        ```
    """
    return load_mlm_example(request.param)


@pytest.fixture(name="eurosat_resnet")
def eurosat_resnet():
    return make_eurosat_resnet()
