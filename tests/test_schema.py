
import pytest


@pytest.fixture
def metadata_json():
    from stac_model.examples import eurosat_resnet
    model_metadata = eurosat_resnet()
    return model_metadata.model_dump_json(indent=2)

def test_model_metadata_json_operations(metadata_json):
    from stac_model.schema import MLModel
    model_metadata = MLModel.model_validate_json(metadata_json)
    assert model_metadata.mlm_name == "Resnet-18 Sentinel-2 ALL MOCO"
