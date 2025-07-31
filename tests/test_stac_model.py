import itertools
from collections.abc import Sized
from typing import Any

import pydantic
import pytest

from stac_model.base import ModelBand, ModelDataVariable
from stac_model.input import InputStructure, ModelInput
from stac_model.output import ModelOutput, ModelResult

ModelClass = type[ModelInput | ModelOutput]


def make_struct(model_class: ModelClass, refs: Sized) -> dict[str, Any]:
    if model_class is ModelInput:
        struct_class = InputStructure
        struct_field = "input"
        struct_xargs = {}
    else:
        struct_class = ModelResult  # type: ignore[assignment]
        struct_field = "result"
        struct_xargs = {"tasks": ["classification"]}
    struct = struct_class(
        shape=[-1, len(refs), 64, 64],
        dim_order=["batch", "channel", "height", "width"],
        data_type="float32",
    )
    return {struct_field: struct, **struct_xargs}


@pytest.mark.parametrize(
    ["model_class", "bands"],
    itertools.product(
        [ModelInput, ModelOutput],
        [
            ["B04", "B03", "B02"],
            [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}],
            [{"name": "NDVI", "format": "rio-calc", "expression": "(B08 - B04) / (B08 + B04)"}],
            [
                "B04",
                {"name": "B03"},
                "B02",
                {"name": "NDVI", "format": "rio-calc", "expression": "(B08 - B04) / (B08 + B04)"},
            ],
        ],
    ),
)
def test_model_band(model_class: ModelClass, bands: list[ModelBand]) -> None:
    struct = make_struct(model_class, bands)
    mlm_object = model_class(
        name="test",
        bands=bands,
        **struct,
    )
    mlm_bands = mlm_object.model_dump()["bands"]
    assert mlm_bands == bands


@pytest.mark.parametrize(
    ["model_class", "bands"],
    itertools.product(
        [ModelInput, ModelOutput],
        [
            [{"name": "test", "expression": "missing-format"}],
            [{"name": "test", "format": "missing-expression"}],
        ],
    ),
)
def test_model_band_format_expression_dependency(model_class: ModelClass, bands: list[ModelBand]) -> None:
    with pytest.raises(pydantic.ValidationError):
        struct = make_struct(model_class, bands)
        ModelInput(
            name="test",
            bands=bands,
            **struct,
        )


@pytest.mark.parametrize(
    "processing_expression",
    [
        None,
        {"format": "test", "expression": "test"},
        [
            {"format": "test", "expression": "test1"},
            {"format": "test", "expression": "test2"},
        ]
    ],
)
def test_model_io_processing_expression_variants(processing_expression):
    model_input = ModelInput(
        name="test",
        bands=[],
        input=InputStructure(
            shape=[-1, 3, 64, 64],
            dim_order=["batch", "channel", "height", "width"],
            data_type="float32",
        ),
        pre_processing_function=processing_expression,
    )
    model_json = model_input.model_dump()
    assert model_json["pre_processing_function"] == processing_expression

    model_output = ModelOutput(
        name="test",
        classes=[],
        tasks={"classification"},
        result=ModelResult(
            shape=[-1, 2, 64, 64],
            dim_order=["batch", "channel", "height", "width"],
            data_type="float32",
        ),
        post_processing_function=processing_expression,
    )
    model_json = model_output.model_dump()
    assert model_json["post_processing_function"] == processing_expression


@pytest.mark.parametrize(
    "variables",
    [
        [
            "temperature_2m",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        [
            {"name": "temperature_2m"},
            {"name": "10m_u_component_of_wind"},
            {"name": "10m_v_component_of_wind"},
        ],
        [
            {"name": "temperature_2m_celsius", "format": "rio-calc", "expression": "temperature_2m + 273.15"},
        ],
        [
            "temperature_2m",
            {"name": "10m_u_component_of_wind"},
            "10m_v_component_of_wind",
            {"name": "temperature_2m_celsius", "format": "rio-calc", "expression": "temperature_2m + 273.15"},
        ],
    ],
)
def test_model_variables(variables: list[ModelDataVariable]) -> None:
    mlm_input = ModelInput(
        name="test",
        variables=variables,
        input=InputStructure(
            shape=[-1, len(variables), 365],
            dim_order=["batch", "variables", "time"],
            data_type="float32",
        ),
    )
    mlm_variables = mlm_input.model_dump()["variables"]
    assert mlm_variables == variables


class Omitted:
    pass


@pytest.mark.parametrize(
    ["model_class", "bands", "variables", "expected_bands", "expected_variables"],
    [  # type: ignore
        (model_cls, *args)
        for model_cls, args in itertools.product(
            [ModelInput, ModelOutput],
            [
                (
                    # explicit empty list should be kept
                    [],
                    [],
                    [],
                    [],
                ),
                (
                    # explicit None should drop the definitions
                    None,
                    None,
                    Omitted,
                    Omitted,
                ),
                (
                    # omitting the properties should default to empty definitions
                    Omitted,
                    Omitted,
                    [],
                    [],
                ),
            ],
        )
    ],
)
def test_model_bands_or_variables_defaults(
    model_class: ModelClass,
    bands: Any,
    variables: Any,
    expected_bands: Any,
    expected_variables: Any,
) -> None:
    mlm_xargs = {}
    if bands is not Omitted:
        mlm_xargs["bands"] = bands
    if variables is not Omitted:
        mlm_xargs["variables"] = variables
    mlm_struct = make_struct(model_class, [1, 2, 3])
    mlm_input = model_class(name="test", **mlm_struct, **mlm_xargs)
    mlm_input_json = mlm_input.model_dump()
    if expected_bands is Omitted:
        assert "bands" not in mlm_input_json
    else:
        mlm_bands = mlm_input_json["bands"]
        assert mlm_bands == expected_bands
    if expected_variables is Omitted:
        assert "variables" not in mlm_input_json
    else:
        mlm_variables = mlm_input_json["variables"]
        assert mlm_variables == expected_variables
