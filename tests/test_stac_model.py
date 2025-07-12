import pydantic
import pytest

from stac_model.input import InputStructure, ModelBand, ModelInput
from stac_model.output import ModelOutput, ModelResult


@pytest.mark.parametrize(
    "bands",
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
)
def test_model_band(bands):
    mlm_input = ModelInput(
        name="test",
        bands=bands,
        input=InputStructure(
            shape=[-1, len(bands), 64, 64],
            dim_order=["batch", "channel", "height", "width"],
            data_type="float32",
        ),
    )
    mlm_bands = mlm_input.dict()["bands"]
    assert mlm_bands == bands


@pytest.mark.parametrize(
    "bands",
    [
        [{"name": "test", "expression": "missing-format"}],
        [{"name": "test", "format": "missing-expression"}],
    ],
)
def test_model_band_format_expression_dependency(bands: list[ModelBand]) -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelInput(
            name="test",
            bands=bands,
            input=InputStructure(
                shape=[-1, len(bands), 64, 64],
                dim_order=["batch", "channel", "height", "width"],
                data_type="float32",
            ),
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
        tasks=["classification"],
        result=ModelResult(
            shape=[-1, 2, 64, 64],
            dim_order=["batch", "channel", "height", "width"],
            data_type="float32",
        ),
        post_processing_function=processing_expression,
    )
    model_json = model_output.model_dump()
    assert model_json["post_processing_function"] == processing_expression
