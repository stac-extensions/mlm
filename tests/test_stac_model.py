import pydantic
import pytest

from stac_model.input import InputStructure, ModelBand, ModelInput


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
            {"name": "NDVI", "format": "rio-calc", "expression": "(B08 - B04) / (B08 + B04)"}
        ],
    ]
)
def test_model_band(bands):
    mlm_input = ModelInput(
        name="test",
        bands=bands,
        input=InputStructure(
            shape=[-1, len(bands), 64, 64],
            dim_order=["batch", "channel", "height", "width"],
            data_type="float32",
        )
    )
    mlm_bands = mlm_input.dict()["bands"]
    assert mlm_bands == bands


@pytest.mark.parametrize(
    "bands",
    [
        [{"name": "test", "expression": "missing-format"}],
        [{"name": "test", "format": "missing-expression"}]
    ]
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
            )
        )
