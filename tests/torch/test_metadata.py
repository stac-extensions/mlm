import pytest

from stac_model.schema import MLModelProperties


@pytest.mark.parametrize(
    "mlm_example",
    ["mlm-metadata.yaml"],
    indirect=True,
)
def test_mlm_metadata_only_yaml_validation(mlm_example):
    MLModelProperties.model_validate(mlm_example)
